# ======================================================================================
# ||                                                                                    ||
# ||                                     IMPORTS                                        ||
# ||                                                                                    ||
# ======================================================================================
import os
import json
import re
import hashlib
import tempfile
import io
import string
import random
from datetime import datetime, timezone, timedelta

# --- Third-party Libraries ---
from dotenv import load_dotenv
import numpy as np
import bcrypt
import pyotp
from pymongo import MongoClient, server_api
from sentence_transformers import SentenceTransformer
import faiss
import dropbox
import PyPDF2

# --- AI SDKs ---
from together import Together
from openai import OpenAI
import google.generativeai as genai


import requests
import base64

# ======================================================================================
# ||                                                                                    ||
# ||                         ENVIRONMENT & GLOBAL CONFIGURATION                           ||
# ||                                                                                    ||
# ======================================================================================
load_dotenv()

# --- API Keys & Secrets ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MongoDB") or os.getenv("MONGO_URI")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

# --- MongoDB Collection & File Names ---
MONGO_DB_NAME = "IntelLawDB_Gradio"
MAIN_CONFIG_COLLECTION_NAME = "main_app_config"
# NEW: Collection for the full scraper payload with categories and schedules
SCRAPER_PAYLOAD_COLLECTION_NAME = "scraper_payload_config" 
ADMIN_USERS_COLLECTION_NAME = "admin_users"
FAISS_COLLECTION_NAME = "faiss_index_store"
TEXT_STORE_COLLECTION_NAME = "text_content_store"
INDEX_FILE_DROPBOX = "/faiss_index.index"
TEXT_FILE_DROPBOX = "/text_store.json"
TOKEN_FILE = "dropbox_token.json"

# --- User Status Constants ---
STATUS_PENDING_EMAIL_CONFIRMATION = "pending_email_confirmation"
STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "suspended"

# --- AI Model Definitions ---
AVAILABLE_MODELS_DICT = {
    "gemini-2.5-flash": {"price": "Custom", "type": "gemini", "name": "Gemini 2.5 Flash"},
    "openai-gpt-4o": {"price": "Custom", "type": "openai", "name": "OpenAI GPT-4o"},
    "meta-llama/Llama-3-70B-Instruct-hf": {"price": "$0.90", "type": "together", "name": "Llama3 70B Instruct (HF)"},
    "meta-llama/Llama-3-8B-Instruct-hf": {"price": "$0.20", "type": "together", "name": "Llama3 8B Instruct (HF)"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.80", "type": "together", "name": "WizardLM-2 8x22B"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together", "name": "Mixtral 8x22B Instruct"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together", "name": "Hermes-2 Mixtral DPO"},
}
AVAILABLE_MODELS_NAMES = sorted([details['name'] for details in AVAILABLE_MODELS_DICT.values()])
MODEL_NAME_TO_ID_MAP = {details['name']: model_id for model_id, details in AVAILABLE_MODELS_DICT.items()}
MODEL_ID_TO_NAME_MAP = {v: k for k, v in MODEL_NAME_TO_ID_MAP.items()}

# --- Global Backend State Variables ---
gemini_model_genai = None
together_client = None
openai_client = None
dbx = None
mongo_client_instance = None
mongo_db_obj = None
auth_mongo_db_obj = None
embedding_model = None
faiss_index = None
text_store = []
BACKEND_INITIAL_LOAD_MSG = "Backend not initialized."

# ======================================================================================
# ||                                                                                    ||
# ||                            AUTHENTICATION & USER MGMT                              ||
# ||                                                                                    ||
# ======================================================================================

def generate_otp(length=6):
    """Generates a random numerical OTP."""
    return "".join(random.choices(string.digits, k=length))

def hash_password(password: str) -> bytes:
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(plain_password: str, hashed_password_bytes: bytes) -> bool:
    """Verifies a plain password against a bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password_bytes)

def get_auth_db():
    """Initializes and/or returns the authentication database object."""
    global auth_mongo_db_obj, mongo_client_instance
    if auth_mongo_db_obj is None:
        if mongo_client_instance is None:
            initialize_mongodb_client()
        if mongo_client_instance is not None:
            auth_mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] 
    return auth_mongo_db_obj

def set_new_password(email: str, new_plain_password: str) -> tuple[bool, str]:
    """
    Finds a user by email and updates their password.
    Hashes the new password before storing.
    """
    db = get_auth_db()
    if db is None:
        return False, "Database not connected."

    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower()
    
    # Check if user exists
    user = users_collection.find_one({"email": email_lower})
    if not user:
        return False, "User not found."

    try:
        new_hashed_password = hash_password(new_plain_password)
        result = users_collection.update_one(
            {"email": email_lower},
            {"$set": {
                "password": new_hashed_password,
                "updated_at": datetime.now(timezone.utc)
            }}
        )
        if result.modified_count > 0:
            print(f"Password updated successfully for {email_lower}")
            return True, "Password updated successfully."
        else:
            # This might happen if the new password hashes to the same as the old one
            print(f"Password for {email_lower} was not modified (already the same?).")
            return True, "Password updated."

    except Exception as e:
        print(f"ERROR setting new password for {email_lower}: {e}")
        return False, "An internal error occurred while updating the password."

# ======================================================================================
# ||                                                                                    ||
# ||                                 MAIN APP CONFIG                                    ||
# ||                                                                                    ||
# ======================================================================================

def get_default_main_config():
    """Returns the default structure for the main application config (model config only)."""
    default_model_id = None
    if AVAILABLE_MODELS_NAMES:
        first_model_name = AVAILABLE_MODELS_NAMES[0]
        default_model_id = MODEL_NAME_TO_ID_MAP.get(first_model_name)

    return {
        "model_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "system_prompt": "You are a helpful assistant. Answer questions based on the provided context.",
            "vary_temperature": True,
            "vary_top_p": False,
            "selected_models": [default_model_id] if default_model_id else []
        }
        # cron_schedule is removed from here as it's now per category in scraper_payload_config
    }

def backend_get_initial_main_config():
    """Loads the main config from DB. If it doesn't exist, creates and returns the default."""
    global mongo_db_obj
    if mongo_db_obj is None:
        print("WARNING: MongoDB not initialized. Returning default main config without saving.")
        return get_default_main_config()
    
    try:
        config_doc = mongo_db_obj[MAIN_CONFIG_COLLECTION_NAME].find_one({"_id": "singleton_config"})
        if config_doc:
            config_doc.pop('_id', None)
            return config_doc
        else:
            print("INFO: No main config found in DB. Creating and saving default config.")
            default_config = get_default_main_config()
            mongo_db_obj[MAIN_CONFIG_COLLECTION_NAME].insert_one({"_id": "singleton_config", **default_config})
            return default_config
    except Exception as e:
        print(f"ERROR: Could not load main config from DB: {e}. Returning default.")
        return get_default_main_config()

def backend_update_main_config(updated_config: dict):
    """Saves the provided main configuration object to the database (only model config related)."""
    global mongo_db_obj
    if mongo_db_obj is None:
        return False, "Database not connected. Config not saved.", updated_config
    try:
        # Ensure only model_config part is handled if needed, or save the whole dict if structure is always fixed.
        # For this setup, we assume updated_config passed from UI only contains valid main config fields.
        mongo_db_obj[MAIN_CONFIG_COLLECTION_NAME].update_one(
            {"_id": "singleton_config"}, {"$set": updated_config}, upsert=True
        )
        return True, "Configuration saved successfully!", updated_config
    except Exception as e:
        print(f"ERROR: Could not save main config to DB: {e}")
        return False, f"Error saving configuration: {e}", updated_config

# ======================================================================================
# ||                                                                                    ||
# ||                              SCRAPER SOURCE CONFIG                                 ||
# ||                                                                                    ||
# ======================================================================================

def get_default_scraper_payload():
    """Returns the default structure for the scraper payload with categories."""
    return {
        "categories": [
            {
                "name": "GDBR_Example",
                "schedule": {
                    "minute": "*/10",
                    "hour": "*",
                    "day_of_month": "*",
                    "month": "*",
                    "day_of_week": "*"
                },
                "links": [
                    {
                        "url": "https://www.imy.se/tillsyner/?query=&page=1",
                        "scraper_type": "pdf"
                    },
                    {
                        "url": "https://eur-lex.europa.eu/legal-content/SV/TXT/HTML/?uri=CELEX:32016R0679",
                        "scraper_type": "text"
                    }
                ]
            }
        ]
    }

def load_scraper_payload_from_db():
    """Loads the entire scraper payload (including categories) from MongoDB."""
    global mongo_db_obj
    if mongo_db_obj is None: 
        print("WARNING: MongoDB not initialized. Returning empty scraper payload.")
        return {"categories": []}
    try:
        payload_doc = mongo_db_obj[SCRAPER_PAYLOAD_COLLECTION_NAME].find_one({"_id": "singleton_scraper_payload"})
        return payload_doc if payload_doc and "categories" in payload_doc else {"categories": []}
    except Exception as e:
        print(f"ERROR: Could not load scraper payload from DB: {e}")
        return {"categories": []}

def save_scraper_payload_to_db(payload_dict: dict):
    """Saves the entire scraper payload dictionary to MongoDB."""
    global mongo_db_obj
    if mongo_db_obj is None:
        return False, "Database not connected."
    try:
        mongo_db_obj[SCRAPER_PAYLOAD_COLLECTION_NAME].update_one(
            {"_id": "singleton_scraper_payload"}, {"$set": payload_dict}, upsert=True
        )
        return True, "Scraper configuration saved."
    except Exception as e:
        print(f"ERROR: Could not save scraper payload to DB: {e}")
        return False, f"Error saving scraper configuration: {e}"

def backend_get_current_scraper_payload():
    """
    Retrieves the current scraper payload from the DB. If no payload exists,
    it initializes with a default and saves it.
    """
    current_payload = load_scraper_payload_from_db()
    if not current_payload.get("categories"): # If DB is empty or 'categories' is missing/empty
        print("INFO: No scraper payload found or categories empty in DB. Initializing with default.")
        default_payload = get_default_scraper_payload()
        success, msg = save_scraper_payload_to_db(default_payload)
        if success:
            print("INFO: Default scraper payload saved.")
            return default_payload
        else:
            print(f"ERROR: Failed to save default scraper payload: {msg}")
            return {"categories": []} # Fallback to empty if save fails
    return current_payload

def backend_add_update_scraper_category(category_name: str, schedule: dict, links: list):
    """
    Adds a new category or updates an existing one in the scraper payload.
    Returns: (success_bool, message_str, updated_payload_dict)
    """
    current_payload = backend_get_current_scraper_payload()
    categories = current_payload.get("categories", [])
    
    # Validate category name
    if not category_name or not category_name.strip():
        return False, "Category name cannot be empty.", current_payload
    
    # Check for duplicate name if adding a new category
    is_new_category = True
    for i, cat in enumerate(categories):
        if cat["name"] == category_name:
            is_new_category = False
            break

    # Validate schedule and links before updating/adding
    # Schedule validation (basic)
    cron_parts = [schedule.get("minute", "*"), schedule.get("hour", "*"), 
                  schedule.get("day_of_month", "*"), schedule.get("month", "*"), 
                  schedule.get("day_of_week", "*")]
    if not all(isinstance(p, str) for p in cron_parts):
        return False, "Schedule parts must be strings.", current_payload
    
    # Links validation
    if not isinstance(links, list):
        return False, "Links must be a list.", current_payload
    for link in links:
        if not isinstance(link, dict) or "url" not in link or "scraper_type" not in link:
            return False, "Each link must be an object with 'url' and 'scraper_type' fields.", current_payload
        if not isinstance(link["url"], str) or not link["url"].strip():
            return False, "Link 'url' cannot be empty.", current_payload
        if link["scraper_type"] not in ["pdf", "text"]:
            return False, f"Invalid scraper_type: '{link['scraper_type']}'. Must be 'pdf' or 'text'.", current_payload
    
    new_category_entry = {
        "name": category_name,
        "schedule": schedule,
        "links": links
    }

    if is_new_category:
        categories.append(new_category_entry)
        msg_prefix = f"Category '{category_name}' added."
    else:
        # Update the existing category (at index i)
        categories[i] = new_category_entry
        msg_prefix = f"Category '{category_name}' updated."

    current_payload["categories"] = categories
    success, save_msg = save_scraper_payload_to_db(current_payload)
    if success:
        return True, f"{msg_prefix} {save_msg}", current_payload
    else:
        return False, f"{msg_prefix} Failed to save to DB: {save_msg}", current_payload

def backend_delete_scraper_category(category_name: str):
    """
    Deletes a category from the scraper payload.
    Returns: (success_bool, message_str, updated_payload_dict)
    """
    current_payload = backend_get_current_scraper_payload()
    categories = current_payload.get("categories", [])
    
    original_len = len(categories)
    updated_categories = [cat for cat in categories if cat["name"] != category_name]

    if len(updated_categories) == original_len:
        return False, f"Category '{category_name}' not found.", current_payload
    
    current_payload["categories"] = updated_categories
    success, save_msg = save_scraper_payload_to_db(current_payload)
    if success:
        return True, f"Category '{category_name}' deleted.", current_payload
    else:
        return False, f"Failed to delete category '{category_name}': {save_msg}", current_payload


# ======================================================================================
# ||                                                                                    ||
# ||                      REMOTE SCRAPER TRIGGERING                                     ||
# ||                                                                                    ||
# ======================================================================================

def trigger_remote_scraper():
    """
    Fetches the current scraper config from MongoDB and sends it to the remote
    web crawler API to initiate a scraping job.
    
    Returns:
        tuple[bool, str]: A tuple containing a success boolean and a status message.
    """
    # 1. Get credentials and API URL from environment variables
    CRAWLER_API_URL = os.getenv("CRAWLER_API_URL")
    CRAWLER_USERNAME = os.getenv("CRAWLER_USERNAME")
    CRAWLER_PASSWORD = os.getenv("CRAWLER_PASSWORD")

    if not all([CRAWLER_API_URL, CRAWLER_USERNAME, CRAWLER_PASSWORD]):
        error_msg = "Crawler API credentials (URL, Username, Password) are not fully configured in environment variables."
        print(f"ERROR: {error_msg}")
        return False, error_msg

    # 2. Get the payload from the database
    # This uses your existing function to get the latest saved configuration.
    payload = backend_get_current_scraper_payload()
    if not payload or not payload.get("categories"):
        return False, "No scraper configuration found in the database. Please configure sources first."

    # 3. Prepare Basic Authentication headers
    auth_string = f"{CRAWLER_USERNAME}:{CRAWLER_PASSWORD}"
    base64_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {base64_auth_string}",
        "Content-Type": "application/json"
    }
    
    # The remote API expects the payload in a list format: [payload]
    request_body = [payload]

    print(f"INFO: Triggering remote scraper at {CRAWLER_API_URL} with {len(payload.get('categories', []))} categories.")

    # 4. Send the POST request
    try:
        # Using a long timeout because scraping can take time
        response = requests.post(CRAWLER_API_URL, json=request_body, headers=headers, timeout=300)

        # 5. Handle the response
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        
        response_data = response.json()
        result_count = len(response_data.get("results", []))
        message = f"Scraper run initiated successfully. The API processed {result_count} configurations."
        print(f"SUCCESS: {message}")
        print(f"API Response: {response_data}")
        return True, message

    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error occurred: {http_err} - Response: {http_err.response.text}"
        print(f"ERROR: {error_message}")
        return False, error_message
    except requests.exceptions.RequestException as req_err:
        error_message = f"An error occurred during the request: {req_err}"
        print(f"ERROR: {error_message}")
        return False, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"ERROR: {error_message}")
        return False, error_message


def backend_run_scraper_now():
    """
    A wrapper function for the Gradio UI. It calls the main scraper trigger
    function and returns a formatted status message for display.
    """
    print("INFO: Manual scraper run triggered from Gradio UI.")
    
    # Call the core function that does the work
    success, message = trigger_remote_scraper()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if success:
        status_prefix = "✅ Success:"
    else:
        status_prefix = "❌ Error:"
        
    # Format a clear message for the UI
    final_message = f"[{timestamp}] {status_prefix}\n{message}"
    
    return final_message


# ======================================================================================
# ||                                                                                    ||
# ||                       INITIALIZATION & DATA PERSISTENCE                            ||
# ||                                                                                    ||
# ======================================================================================

def initialize_mongodb_client():
    global mongo_client_instance, mongo_db_obj, auth_mongo_db_obj
    if MONGO_URI and mongo_client_instance is None:
        try:
            mongo_client_instance = MongoClient(MONGO_URI, server_api=server_api.ServerApi('1'))
            mongo_client_instance.admin.command('ping') # Verify connection
            mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] # For app data (FAISS, text_store)
            auth_mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] # For auth data (admin_users)
                                                                  # Using same DB, but conceptually could be different
            print("MongoDB client initialized successfully.")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            mongo_client_instance = None; mongo_db_obj = None; auth_mongo_db_obj = None
    elif not MONGO_URI:
        print("Warning: MONGO_URI not set. MongoDB features disabled.")
    # else:
        # print("MongoDB client already initialized or MONGO_URI not set.")

def initialize_dropbox_client():
    global dbx
    if not (DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        print("Warning: Dropbox credentials not fully set. Dropbox features disabled.")
        dbx = None; return
    try:
        # get_valid_access_token() is an external helper, assumed to exist or mock
        # For simplicity in this example, let's assume direct token or a placeholder
        # In a real app, this would involve OAuth flow and storing/refreshing tokens.
        # For this backend, assuming dbx is initialized from Flask part or by direct token.
        # Placeholder for demonstration, actual implementation needs real token refresh logic
        # For now, let's just create a dummy client if no actual token management is provided
        # if access_token:
        #     dbx = dropbox.Dropbox(access_token)
        #     print("Dropbox client initialized successfully.")
        # else:
        #     print("Failed to obtain Dropbox access token. Dropbox client not initialized.")
        dbx = dropbox.Dropbox("dummy_access_token_if_needed") # Replace with actual token retrieval
        print("Dropbox client initialized successfully (Placeholder).")
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        dbx = None

def save_data_to_selected_db(selected_db):
    global faiss_index, text_store, dbx, mongo_db_obj

    if selected_db == "Dropbox" and dbx is None: initialize_dropbox_client()
    if selected_db == "MongoDB" and mongo_db_obj is None: initialize_mongodb_client() 

    index_to_save = faiss_index
    if index_to_save is None or embedding_model is None:
        print("Warning: FAISS index or embedding model is None. Cannot save empty or uninitialized index.")
        if embedding_model and index_to_save is None:
            index_to_save = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            print("Created new empty FAISS index for saving.")
        else:
            return

    if selected_db == "Dropbox":
        if dbx is None: print("Dropbox not initialized for saving."); return
        try:
            if index_to_save.ntotal == 0:
                print("Skipping FAISS index save to Dropbox as it's empty.")
            else:
                temp_idx_file = "temp_faiss_to_dropbox.index"
                faiss.write_index(index_to_save, temp_idx_file)
                with open(temp_idx_file, "rb") as f: dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
                os.remove(temp_idx_file)
            
            text_json = json.dumps(text_store, ensure_ascii=False, indent=4).encode('utf-8')
            dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            print(f"Data saved to Dropbox. Index size: {index_to_save.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to Dropbox: {e}")
    elif selected_db == "MongoDB":
        if mongo_db_obj is None: print("MongoDB not initialized for saving app data."); return
        try:
            if index_to_save.ntotal == 0:
                 print("Skipping FAISS index save to MongoDB as it's empty.")
                 mongo_db_obj[FAISS_COLLECTION_NAME].delete_one({"_id": "main_faiss_index"})
            else:
                temp_idx_file = "temp_faiss_to_mongo.idx"
                faiss.write_index(index_to_save, temp_idx_file)
                with open(temp_idx_file, "rb") as f: index_bytes = f.read()
                os.remove(temp_idx_file)
                mongo_db_obj[FAISS_COLLECTION_NAME].update_one({"_id": "main_faiss_index"}, {"$set": {"index_data": index_bytes}}, upsert=True)

            mongo_db_obj[TEXT_STORE_COLLECTION_NAME].delete_many({})
            if text_store: mongo_db_obj[TEXT_STORE_COLLECTION_NAME].insert_many(text_store)
            print(f"Data saved to MongoDB. Index size: {index_to_save.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to MongoDB: {e}")


def load_data_from_selected_db(selected_db):
    global faiss_index, text_store, embedding_model, dbx, mongo_db_obj
    if embedding_model is None:
        print("Error: Embedding model not initialized. Cannot load data.")
        return "Embedding model not initialized. Load failed."

    # Initialize fresh, empty structures
    # Ensure embedding_model.get_sentence_embedding_dimension() returns a valid integer
    try:
        dimension = embedding_model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Invalid embedding dimension: {dimension}")
        faiss_index = faiss.IndexFlatL2(dimension)
    except Exception as e:
        print(f"Critical error initializing FAISS index with dimension: {e}")
        return f"Critical error initializing FAISS index: {e}. Load failed."

    text_store = []
    alert_msg = ""
    temp_idx_file_path = None # To ensure it's defined for finally block

    if selected_db == "Dropbox":
        # ... (your existing Dropbox code - ensure it also handles faiss.read_index errors robustly) ...
        if dbx is None: initialize_dropbox_client()
        if dbx is None: alert_msg = "Dropbox not initialized. Cannot load."; return alert_msg
        try:
            temp_idx_file_path = "temp_faiss_from_dropbox.index"
            # Replace with actual Dropbox download function if `dbx` is properly initialized
            # For demonstration, assume files are empty or don't exist
            # _, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
            # with open(temp_idx_file_path, "wb") as f: f.write(res_index.content)
            # loaded_index = faiss.read_index(temp_idx_file_path)
            # if loaded_index.d == faiss_index.d:
            #     faiss_index = loaded_index
            # else:
            #     alert_msg += f"Warning: Dropbox index dimension mismatch ({loaded_index.d} vs {faiss_index.d}). Not loading. "
            
            # _, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
            # text_store = json.loads(res_text.content.decode('utf-8'))
            alert_msg += "Dropbox functions are stubbed. No data loaded from Dropbox."
        except dropbox.exceptions.ApiError as e:
            if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_path() and e.error.get_path().is_not_found():
                alert_msg += "No existing data on Dropbox. Initialized empty store."
            else: alert_msg += f"Dropbox API error: {e}"
        except RuntimeError as faiss_error: # Catch FAISS read errors
            alert_msg += f"Error reading FAISS index from Dropbox: {faiss_error}. Index likely corrupt. Re-initializing. "
            # faiss_index remains the newly initialized empty one
        except Exception as e: alert_msg += f"Error loading from Dropbox: {e}"
        finally:
            if temp_idx_file_path and os.path.exists(temp_idx_file_path):
                os.remove(temp_idx_file_path)


    elif selected_db == "MongoDB":
        if mongo_db_obj is None: initialize_mongodb_client()
        if mongo_db_obj is None: alert_msg = "MongoDB not initialized for app data. Cannot load."; return alert_msg
        
        index_doc = None # Define before try block
        try:
            index_doc = mongo_db_obj[FAISS_COLLECTION_NAME].find_one({"_id": "main_faiss_index"})
            if index_doc and "index_data" in index_doc and index_doc["index_data"]: # Check if index_data exists and is not empty
                temp_idx_file_path = "temp_faiss_from_mongo.idx"
                with open(temp_idx_file_path, "wb") as f: f.write(index_doc["index_data"])
                
                # Before attempting to read, check if file has content
                if os.path.getsize(temp_idx_file_path) == 0:
                    raise RuntimeError("Temporary FAISS index file is empty after writing from MongoDB data.")

                loaded_index = faiss.read_index(temp_idx_file_path) # This is where the error occurs
                if loaded_index.d == faiss_index.d: # faiss_index is the newly initialized one
                    faiss_index = loaded_index
                    # alert_msg += f"FAISS index loaded from MongoDB. Dim: {faiss_index.d}, Vectors: {faiss_index.ntotal}. " # Becomes too verbose
                else:
                    alert_msg += f"Warning: MongoDB index dimension mismatch ({loaded_index.d} vs {faiss_index.d}). Not loading. "
            else:
                 alert_msg += "No FAISS index data found or data is empty in MongoDB. "
        
        except RuntimeError as faiss_error: # Catch FAISS-specific runtime errors
            alert_msg += f"Error reading FAISS index from MongoDB data: {faiss_error}. Index is likely corrupt. "
            if index_doc and "_id" in index_doc: # If we know the document ID that caused the error
                try:
                    delete_result = mongo_db_obj[FAISS_COLLECTION_NAME].delete_one({"_id": index_doc["_id"]})
                    if delete_result.deleted_count > 0:
                        alert_msg += "Corrupted FAISS index document deleted from MongoDB. "
                    else:
                        alert_msg += "Attempted to delete corrupted index, but document not found (or already deleted). "
                except Exception as db_del_err:
                    alert_msg += f"Error trying to delete corrupted index from MongoDB: {db_del_err}. "
            else:
                alert_msg += "Cannot identify specific MongoDB document for corrupted index to delete. "
            # faiss_index remains the newly initialized empty one
        except Exception as e: # Catch other general errors during index loading phase
            alert_msg += f"General error during MongoDB FAISS index loading: {e}. "
        finally:
            if temp_idx_file_path and os.path.exists(temp_idx_file_path):
                os.remove(temp_idx_file_path)

        # Load text_store regardless of index success/failure
        try:
            text_docs = list(mongo_db_obj[TEXT_STORE_COLLECTION_NAME].find({}))
            text_store = [{k: v for k, v in doc.items() if k != '_id'} for doc in text_docs]
            alert_msg += f"Loaded {len(text_store)} text items from MongoDB. "
        except Exception as e:
            alert_msg += f"Error loading text_store from MongoDB: {e}. "
            text_store = [] # Ensure text_store is empty on error

        # Final status message construction
        if faiss_index.ntotal > 0:
            alert_msg += f"Final Index: {faiss_index.ntotal} vectors. "
        else:
            alert_msg += f"Final Index: Empty. "

        if not (index_doc and "index_data" in index_doc and index_doc["index_data"]) and not text_docs:
             if "No FAISS index data found" in alert_msg and "0 text items" in alert_msg: # Check if already handled
                pass # Message likely already comprehensive
             else:
                alert_msg += "Initialized empty store as no data found in MongoDB. "


    # Consistency checks (these are good to keep)
    if faiss_index.ntotal > 0 and not text_store and "corrupt" not in alert_msg.lower():
        alert_msg += " Warning: Index has vectors but text store is empty. Data might be inconsistent. Clearing index to be safe."
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    elif faiss_index.ntotal == 0 and text_store and ("corrupt" not in alert_msg.lower() and "mismatch" not in alert_msg.lower()):
        alert_msg += " Warning: Text store loaded but index is empty/failed to load. Consider re-indexing. "

    final_status_message = f"DB Load Status ({selected_db}): {alert_msg.strip()}"
    print(final_status_message)
    return final_status_message
# ======================================================================================
# ||                                                                                    ||
# ||                         CORE AI, RAG & PDF PROCESSING                              ||
# ||                                                                                    ||
# ======================================================================================

def chunk_text(text, chunk_size=400, min_chunk_length=20):
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        temp_chunk = ""
        for sentence in sentences:
            if len(temp_chunk) + len(sentence) < chunk_size:
                temp_chunk += sentence + " "
            else:
                cleaned_chunk = temp_chunk.strip()
                if len(cleaned_chunk) >= min_chunk_length:
                    chunks.append(cleaned_chunk)
                temp_chunk = sentence + " "
        cleaned_chunk = temp_chunk.strip()
        if len(cleaned_chunk) >= min_chunk_length:
            chunks.append(cleaned_chunk)
    return chunks

def extract_text_from_pdf_bytes(pdf_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes)); text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        text += page_text + "\n" if page_text else ""
    return text

def process_and_add_pdf_core(pdf_bytes, file_name, selected_db):
    global text_store, faiss_index, embedding_model
    if embedding_model is None: return "Embedding model not initialized.", False
    if faiss_index is None: 
        if embedding_model: faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        else: return "FAISS index not initialized and embedding model missing.", False

    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    if any(item.get('file_hash') == file_hash for item in text_store if isinstance(item, dict)):
        return f"File '{file_name}' (hash: {file_hash[:7]}) seems to already exist based on hash.", False
    
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not raw_text.strip():
        return f"No text could be extracted from '{file_name}'.", False
        
    chunks = chunk_text(raw_text)
    if not chunks:
        return f"Could not break '{file_name}' into processable chunks.", False

    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
    embeddings_np = np.array(embeddings).astype("float32")
    if embeddings_np.shape[0] > 0: faiss_index.add(embeddings_np)
    
    for chunk_text_content in chunks:
        text_store.append({"text": chunk_text_content, "file_name": file_name, "file_hash": file_hash})
    return f"Processed and added '{file_name}'. Chunks: {len(chunks)}, Index size: {faiss_index.ntotal}", True

def retrieve_context_from_db(query, top_k=5):
    if embedding_model is None or faiss_index is None or faiss_index.ntotal == 0:
        return "No documents are available for context retrieval."
    try:
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)
        _, indices = faiss_index.search(np.array(query_embedding).astype("float32"), top_k)
        retrieved_texts = [text_store[i]["text"] for i in indices[0] if 0 <= i < len(text_store)]
        return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."
    except Exception as e:
        print(f"ERROR during context retrieval: {e}")
        return "An error occurred while retrieving context."

def generate_response_gemini(prompt, context, temp, top_p, system_prompt):
    if not gemini_model_genai: return "Gemini client not initialized."
    input_parts = [system_prompt + "\nContext: " + context, "Question: " + prompt]
    config = genai.GenerationConfig(max_output_tokens=2048, temperature=temp, top_p=top_p)
    try: response = gemini_model_genai.generate_content(input_parts, generation_config=config); return response.text
    except Exception as e: return f"Gemini Error: {e}"

def generate_response_together_ai(prompt, context, model_id, temp, top_p, system_prompt):
    if not together_client: return "TogetherAI client not initialized."
    try:
        response = together_client.chat.completions.create(
            model=model_id, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e: return f"TogetherAI Error ({model_id}): {e}"

def generate_response_openai_api(prompt, context, temp, top_p, system_prompt): 
    if not openai_client: return "OpenAI client not initialized."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e: return f"OpenAI Error: {e}"

# ======================================================================================
# ||                                                                                    ||
# ||                        UI-FACING BACKEND CALLBACKS                                 ||
# ||                                                                                    ||
# ======================================================================================

def get_unique_filenames_from_text_store():
    if not text_store: return []
    return sorted(list({item["file_name"] for item in text_store if "file_name" in item}))

def switch_db_backend(selected_db_val, current_selected_db_state_value):
    if selected_db_val == current_selected_db_state_value:
        return {
                "selected_db": selected_db_val,
                "status": f"Already using {selected_db_val}."
            }
    status = load_data_from_selected_db(selected_db_val)
    return {
            "selected_db": selected_db_val,
            "status": status,
            "text_store_choices": get_unique_filenames_from_text_store(),
            "text_store_value": []
        }

def handle_pdf_upload_backend(files_obj_list, selected_db_from_state):
    if not files_obj_list:
        return {
            "message": "No files were uploaded."
        }
    alerts = [process_and_add_pdf_core(open(f.name, "rb").read(), os.path.basename(f.name), selected_db_from_state)[0] for f in files_obj_list]
    save_data_to_selected_db(selected_db_from_state)
    return {
            "message": "Files processed and added.",
            "status": "\n".join(alerts),
            "text_store_choices": get_unique_filenames_from_text_store(),
            "text_store_value": []
        }

def delete_files_backend(filenames_to_delete, selected_db):
    global text_store, faiss_index
    if not filenames_to_delete:
        return {
            "message": "No files selected for deletion."
        }
    initial_len = len(text_store)
    indices_to_remove = {i for i, item in enumerate(text_store) if item.get("file_name") in filenames_to_delete}
    if not indices_to_remove:
        return {
            "message": "No matching files found in the text store."
        }
    
    if faiss_index and faiss_index.ntotal > 0:
        new_embeddings = np.array([embedding_model.encode([text_store[i]["text"]])[0] for i in range(len(text_store)) if i not in indices_to_remove]).astype("float32")
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        if new_embeddings.shape[0] > 0:
            faiss_index.add(new_embeddings)

    text_store = [item for i, item in enumerate(text_store) if i not in indices_to_remove]
    save_data_to_selected_db(selected_db)
    return {
            "message": f"Deleted {len(filenames_to_delete)} file(s) and their {len(indices_to_remove)} chunks.",
            "updated_filenames": get_unique_filenames_from_text_store()
        }
def backend_apply_uploaded_main_config(config_file_obj):
    if config_file_obj is None:
        return "No config file uploaded.", False, backend_get_initial_main_config()
    try:
        with open(config_file_obj.name, 'r', encoding='utf-8') as f:
            new_config = json.load(f)
        # Validate structure: only model_config expected here
        if "model_config" not in new_config:
            raise ValueError("Uploaded config is missing 'model_config' key or malformed. Only model config supported here.")
        
        # Merge with current config to preserve other potential top-level keys if any, though none currently
        current_main_config = backend_get_initial_main_config()
        current_main_config["model_config"] = new_config["model_config"] # Overwrite model_config
        
        success, msg, final_config = backend_update_main_config(current_main_config)
        return "Config applied from file." if success else msg, success, final_config
    except Exception as e:
        return f"Error applying config: {e}", False, backend_get_initial_main_config()

def backend_generate_main_config_for_download(current_main_config: dict):
    """Generates a temporary JSON file of the main application config (model_config only) for download."""
    try:
        # Filter to include only 'model_config' for download, as 'cron_schedule' is moved.
        downloadable_config = {"model_config": current_main_config.get("model_config", {})}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding='utf-8') as tmp_file:
            json.dump(downloadable_config, tmp_file, indent=2)
            return tmp_file.name
    except Exception as e:
        print(f"ERROR: Could not generate config for download: {e}")
        return None

# NOTE: get_scraper_ui_initial_data is replaced by backend_get_current_scraper_payload
# NOTE: backend_update_scraper_source_in_db and backend_remove_scraper_source_from_db are replaced
# by backend_add_update_scraper_category and backend_delete_scraper_category

def chat_interface_backend(
    user_input: str,
    chat_history: list[list[str | None]] | None,
    selected_db_state_val: str,
    model_config_dict: dict
) -> list[list[str | None]]:

    if not user_input or not user_input.strip():
        return chat_history if chat_history is not None else []

    current_chat_history = chat_history if chat_history is not None else []

    # 1. Retrieve context from the vector database using RAG
    print(f"Retrieving context for query: '{user_input[:50]}...'")
    context_text = retrieve_context_from_db(user_input, top_k=5)

    # 2. Extract model parameters from the passed-in dictionary
    temp_config = model_config_dict.get('temperature', 0.7)
    top_p_config = model_config_dict.get('top_p', 0.9)
    system_prompt_config = model_config_dict.get('system_prompt', "You are a helpful assistant.")
    selected_ai_model_ids = model_config_dict.get('selected_models', [])

    # 3. Determine parameter variations based on checkboxes
    temp_values_to_run = [temp_config]
    if model_config_dict.get('vary_temperature', False) and temp_config > 0.01:
        # Create a list of varied temperatures, clamping values between 0.01 and 1.0
        temp_values_to_run = sorted(list(set([
            round(max(0.01, temp_config * 0.5), 2), 
            temp_config, 
            round(min(1.0, temp_config * 1.5), 2)
        ])))

    top_p_values_to_run = [top_p_config]
    if model_config_dict.get('vary_top_p', False) and top_p_config > 0.01:
        # Create a list of varied top_p values, clamping values between 0.01 and 1.0
        top_p_values_to_run = sorted(list(set([
            round(max(0.01, top_p_config * 0.5), 2), 
            top_p_config,
            round(min(1.0, top_p_config * 1.5), 2)
        ])))

    # 4. Generate responses from all selected models and parameter combinations
    bot_response_content_parts = []
    if not selected_ai_model_ids:
        final_bot_response = "System: No AI model has been selected in the configuration. Please ask an admin to configure one."
    else:
        for model_id in selected_ai_model_ids:
            model_detail = AVAILABLE_MODELS_DICT.get(model_id, {})
            model_type = model_detail.get("type")
            model_display_name = model_detail.get("name", model_id)

            for temp_val in temp_values_to_run:
                for top_p_val in top_p_values_to_run:
                    print(f"Generating response from {model_display_name} (Temp: {temp_val}, Top-P: {top_p_val})")
                    response_content = f"Error: Model type '{model_type}' for '{model_display_name}' is not configured correctly."
                    
                    # Dynamically call the appropriate generation function
                    if model_type == "gemini" and gemini_model_genai:
                        response_content = generate_response_gemini(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "together" and together_client:
                        response_content = generate_response_together_ai(user_input, context_text, model_id, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "openai" and openai_client:
                        response_content = generate_response_openai_api(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    
                    # Format the response with model details
                    model_info_str = f"{model_display_name} (T:{temp_val}, P:{top_p_val})"
                    bot_response_content_parts.append(f"--- {model_info_str} ---\n{response_content}")

                    if not model_config_dict.get('vary_top_p', False):
                        break  # Exit inner loop if not varying top_p
                if not model_config_dict.get('vary_temperature', False):
                    break  # Exit outer loop if not varying temperature

        final_bot_response = "\n\n".join(bot_response_content_parts)
        if not final_bot_response:
            final_bot_response = "System: No responses were generated. This could be due to an issue with the selected AI models or their configurations."

    # 5. Append the user message and final bot response to history
    current_chat_history.append([user_input, final_bot_response])
    return current_chat_history

def get_backend_initial_load_message():
    global BACKEND_INITIAL_LOAD_MSG
    return BACKEND_INITIAL_LOAD_MSG

# ======================================================================================
# ||                                                                                    ||
# ||          REQUIRED IMPORTS & CONSTANTS (Ensure these are at the top)                ||
# ||                                                                                    ||
# ======================================================================================

# --- User Status Constants (should be defined in your file) ---
STATUS_PENDING_EMAIL_CONFIRMATION = "pending_email_confirmation"
STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "suspended"

# This helper function is assumed to exist and work correctly
def get_auth_db():
    # ... your existing logic to get a MongoDB database object ...
    global auth_mongo_db_obj
    return auth_mongo_db_obj

# ======================================================================================
# ||                                                                                    ||
# ||                     AUTHENTICATION & 2FA HELPER FUNCTIONS                          ||
# ||                                                                                    ||
# ======================================================================================

def get_full_user_for_auth(email):
    """
    Fetches the full user document, including sensitive fields like password
    and TOTP secret, for authentication purposes.
    USE WITH CAUTION and only in authentication flows.
    """
    db = get_auth_db()
    if db is None:
        print(f"Auth DB not available when fetching full user for {email}")
        return None
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    return users_collection.find_one({"email": email.lower()})

def get_user_by_email(email):
    db = get_auth_db()
    if db is None:
        print("Error: MongoDB for auth not available. Cannot get user.")
        return None
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    return users_collection.find_one({"email": email})

def get_all_users_from_db():
    # ... (ensure this function correctly fetches all users and converts timestamps to datetime) ...
    # Make sure it returns the 'status' field as is from the DB.
    # The renaming of 'pending' to 'pending_admin_approval' will be reflected here if the DB stores it that way.
    db = get_auth_db()
    if db is None: return []
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    try:
        users_cursor = users_collection.find({})
        users_list = []
        for user_doc in users_cursor:
            user_doc['_id'] = str(user_doc['_id'])
            if 'password' in user_doc: del user_doc['password']
            for ts_field in ["created_at", "updated_at", "last_login_at"]: # Add deleted_at
                if ts_field in user_doc:
                    if isinstance(user_doc[ts_field], (int, float)):
                        user_doc[ts_field] = datetime.fromtimestamp(user_doc[ts_field], timezone.utc)
                    elif isinstance(user_doc[ts_field], datetime) and user_doc[ts_field].tzinfo is None:
                         user_doc[ts_field] = user_doc[ts_field].replace(tzinfo=timezone.utc)
            if 'full_name' not in user_doc or not user_doc['full_name']:
                user_doc['full_name'] = user_doc.get('email', "N/A").split('@')[0]
            if 'totp_secret' in user_doc: del user_doc['totp_secret']
            if 'recovery_codes' in user_doc: del user_doc['recovery_codes']
            if 'used_recovery_codes' in user_doc: del user_doc['used_recovery_codes']
            # You can include 'is_2fa_enabled'
            user_doc['is_2fa_enabled'] = user_doc.get('is_2fa_enabled', False)
            users_list.append(user_doc)
        return users_list
    except Exception as e:
        print(f"Error fetching all users: {e}"); return []

def set_user_totp_secret(email, totp_secret):
    """Stores the TOTP secret for a user (usually before it's fully enabled)."""
    db = get_auth_db()
    if db is None:  # <<< CORRECTED CHECK
        print("Error: Auth DB not available in set_user_totp_secret.")
        return False, "Database connection error. Failed to set TOTP secret."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    try:
        result = users_collection.update_one(
            {"email": email.lower()},
            {"$set": {"totp_secret": totp_secret, "is_2fa_enabled": False, "updated_at": datetime.now(timezone.utc)}}
        )
        if result.modified_count > 0 or result.matched_count > 0 : # matched_count in case it was already set to the same
            return True, "TOTP secret set/updated."
        else: # User not found
            return False, "User not found. Failed to set TOTP secret."
    except Exception as e:
        print(f"Error in set_user_totp_secret for {email}: {e}")
        return False, "Database operation error."

def enable_user_2fa(email):
    """Enables 2FA for the user, stores hashed recovery codes, and marks initial login as complete."""
    db = get_auth_db()
    if db is None:  # <<< CORRECTED CHECK
        print("Error: Auth DB not available in enable_user_2fa.")
        return False, "Database connection error. Failed to enable 2FA."
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower() # Ensure consistent casing
    user = users_collection.find_one({"email": email_lower})

    if not user: # Added check for user existence
        print(f"Error: User {email_lower} not found in enable_user_2fa.")
        return False, "User not found. Cannot enable 2FA."
        
    if not user.get("totp_secret"): # Prerequisite check
        print(f"Error: TOTP secret not set for {email_lower} before enabling 2FA.")
        return False, "Cannot enable 2FA: Critical setup step (TOTP secret) missing."

    try:
        result = users_collection.update_one(
            {"email": email_lower}, # Use email_lower here too
            {"$set": {
                "is_2fa_enabled": True,
                "has_completed_initial_login": True, 
                "updated_at": datetime.now(timezone.utc)
            }}
        )
        if result.modified_count > 0:
            print(f"2FA enabled for user {email_lower}.")
            return True, "2FA enabled successfully."
        elif result.matched_count > 0 and user.get("is_2fa_enabled") is True: # Already enabled with same codes?
            print(f"2FA was already enabled for {email_lower}, considered success.")
            return True, "2FA was already enabled." # Or a more specific message
        else:
            print(f"Failed to enable 2FA for {email_lower}. Matched: {result.matched_count}, Modified: {result.modified_count}")
            return False, "Failed to update record to enable 2FA."
            
    except Exception as e:
        print(f"Database error in enable_user_2fa for {email_lower}: {e}")
        return False, "Database operation error during 2FA enabling."

def verify_totp_code(totp_secret, submitted_code):
    """Verifies a TOTP code against the user's secret."""
    if not totp_secret or not submitted_code:
        return False
    totp = pyotp.TOTP(totp_secret)
    return totp.verify(submitted_code, valid_window=1) # Allow current, previous, and next window (e.g., +/- 30s)

# ======================================================================================
# ||                                                                                    ||
# ||                  NEW USER REGISTRATION & EMAIL VERIFICATION                        ||
# ||                                                                                    ||
# ======================================================================================
def create_user(email, plain_password):
    """
    Creates a new user with 'pending_email_confirmation' status,
    generates an OTP, and stores it.
    Returns: (success_bool, message_str, otp_str_or_None)
    """
    db = get_auth_db()
    if db is None:
        return False, "Database error, please try again later.", None
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower()
    existing_user = users_collection.find_one({"email": email_lower})
    current_dt = datetime.now(timezone.utc)
    
    otp = generate_otp()
    otp_expiry = current_dt + timedelta(minutes=10) # OTP valid for 10 minutes

    if existing_user:
        if existing_user.get("status") == STATUS_PENDING_EMAIL_CONFIRMATION:
            # User exists but email not confirmed. Update password, regenerate OTP.
            hashed_pass = hash_password(plain_password)
            users_collection.update_one(
                {"email": email_lower},
                {"$set": {
                    "password": hashed_pass,
                    "updated_at": current_dt,
                    "created_at": current_dt, # Optionally refresh for new OTP window
                    "email_otp": otp,          # Update OTP
                    "otp_expires_at": otp_expiry # Update OTP expiry
                }}
            )
            print(f"User {email_lower} re-attempted signup. OTP regenerated.")
            return True, "An OTP has been re-sent to your email.", otp
        else: # User exists and is in another state (active, suspended)
            return False, "Email address already registered and confirmed or in another state.", None

    hashed_pass = hash_password(plain_password)
    try:
        user_doc_fields = {
            "email": email_lower,
            "password": hashed_pass,
            "role": "user",
            "status": STATUS_PENDING_EMAIL_CONFIRMATION,
            "created_at": current_dt,
            "updated_at": current_dt,
            "full_name": email_lower.split('@')[0],
            "email_otp": otp,
            "otp_expires_at": otp_expiry,
            "has_completed_initial_login": False, # << NEW
            "is_2fa_enabled": False,       # << NEW: 2FA status
            "totp_secret": None
        }
        users_collection.insert_one(user_doc_fields)
        print(f"User {email_lower} registered with {STATUS_PENDING_EMAIL_CONFIRMATION} status. OTP generated.")
        return True, "User record created. An OTP has been sent to your email.", otp
    except Exception as e:
        print(f"Error creating user {email_lower}: {e}")
        return False, "An error occurred during registration.", None

def verify_otp_and_activate_user(email, submitted_otp):
    """
    Verifies the submitted OTP for the given email and activates the user if valid.
    Returns: (success_bool, message_str)
    """
    db = get_auth_db()
    if db is None: return False, "Database error."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower()
    user = users_collection.find_one({"email": email_lower})

    if not user: return False, "User not found."
    if user.get("status") == STATUS_SUSPENDED: return True, "Account already created."

    if user.get("status") != STATUS_PENDING_EMAIL_CONFIRMATION:
        return False, "Account not awaiting email confirmation."

    stored_otp = user.get("email_otp")
    otp_expires_at = user.get("otp_expires_at") 

    if not stored_otp or not otp_expires_at:
        return False, "OTP not found or has an issue. Please request a new one."
    
    # --- FIX STARTS HERE ---
    # Ensure otp_expires_at is a datetime object and make it timezone-aware (assuming UTC)
    if not isinstance(otp_expires_at, datetime):
        # This case should ideally not happen if you store datetime objects.
        # If it's a string or timestamp, you'd need to parse it first.
        # For example, if it was a Unix timestamp (float):
        # otp_expires_at = datetime.fromtimestamp(otp_expires_at, timezone.utc)
        print(f"ERROR: otp_expires_at for {email_lower} is not a datetime object from DB: {type(otp_expires_at)}")
        return False, "Internal error with OTP expiry format. Please try again."

    if otp_expires_at.tzinfo is None:
        # If it's naive, assume it was stored as UTC and make it UTC-aware
        otp_expires_at = otp_expires_at.replace(tzinfo=timezone.utc)
    # --- FIX ENDS HERE ---

    current_time_utc = datetime.now(timezone.utc) # Get current UTC time once

    if otp_expires_at < current_time_utc:
        # Clear expired OTP
        users_collection.update_one({"_id": user["_id"]}, {"$unset": {"email_otp": "", "otp_expires_at": ""}})
        return False, "OTP has expired. Please request a new one."

    if stored_otp == submitted_otp:
        users_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {"status": STATUS_SUSPENDED, "updated_at": current_time_utc}, # Use consistent current time
                "$unset": {"email_otp": "", "otp_expires_at": ""} 
            }
        )
        return True, "Email confirmed successfully! Your account is now active."
    else:
        return False, "Invalid OTP entered."

def regenerate_otp_for_user(email):
    """
    Regenerates an OTP for a user whose status is 'pending_email_confirmation'.
    Returns: (new_otp_str_or_None, message_str)
    """
    db = get_auth_db()
    if db is None: return None, "Database error."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower()
    
    # Find user and ensure they are in the correct state to receive a new OTP
    user = users_collection.find_one({"email": email_lower})
    if not user:
        return None, "User not found."
    if user.get("status") != STATUS_PENDING_EMAIL_CONFIRMATION:
        return None, "Account is not awaiting email confirmation (e.g., already active or suspended)."

    new_otp = generate_otp()
    new_otp_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    try:
        users_collection.update_one(
            {"_id": user["_id"]}, # Use _id for precision
            {"$set": {
                "email_otp": new_otp,
                "otp_expires_at": new_otp_expiry,
                "updated_at": datetime.now(timezone.utc)
            }}
        )
        print(f"OTP regenerated for {email_lower}.")
        return new_otp, "A new OTP has been generated."
    except Exception as e:
        print(f"Error regenerating OTP for {email_lower}: {e}")
        return None, "Failed to regenerate OTP."


# ======================================================================================
# ||                                                                                    ||
# ||                            ADMIN PANEL USER MANAGEMENT                             ||
# ||                                                                                    ||
# ======================================================================================


def update_user_status_in_db(user_email: str, new_status: str) -> tuple[bool, str]:
    """
    Allows an admin to update a user's status to 'active' or 'suspended'.

    Args:
        user_email: The email of the user to update.
        new_status: The target status (must be 'active' or 'suspended').

    Returns:
        A tuple containing a success boolean and a status message.
    """
    db = get_auth_db()
    if db is None: return False, "Database not connected."
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    admin_allowed_statuses = [STATUS_ACTIVE, STATUS_SUSPENDED]
    
    if new_status not in admin_allowed_statuses:
        return False, f"Invalid target status '{new_status}' for admin action."

    try:
        result = users_collection.update_one(
            {"email": user_email.lower()},
            {"$set": {"status": new_status, "updated_at": datetime.now(timezone.utc)}}
        )
        if result.matched_count == 0:
            return False, f"User '{user_email}' not found."
        if result.modified_count == 0:
            return True, f"User '{user_email}' was already set to '{new_status}'."
        return True, f"User '{user_email}' status successfully updated to '{new_status}'."
    except Exception as e:
        print(f"ERROR updating user status for {user_email}: {e}")
        return False, "An internal database error occurred."

def hard_delete_user_from_db(user_email: str) -> tuple[bool, str]:
    """
    Permanently deletes a user from the database. This action is irreversible.

    Args:
        user_email: The email of the user to permanently delete.

    Returns:
        A tuple containing a success boolean and a status message.
    """
    db = get_auth_db()
    if db is None:
        return False, "Database not connected."
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    try:
        result = users_collection.delete_one({"email": user_email.lower()})
        if result.deleted_count == 1:
            print(f"User '{user_email}' has been permanently deleted.")
            return True, f"User '{user_email}' has been permanently deleted."
        else:
            return False, f"User '{user_email}' not found for deletion."
    except Exception as e:
        print(f"ERROR during hard delete for user {user_email}: {e}")
        return False, "An internal database error occurred."
    
# ======================================================================================
# ||                                                                                    ||
# ||                              BACKEND INITIALIZATION                                ||
# ||                                                                                    ||
# ======================================================================================

def create_admin_user_if_not_exists(email, plain_password, role="admin"):
    db = get_auth_db()
    if db is None:
        print("Error: MongoDB for auth not available. Cannot create/update admin user.")
        return False
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    
    user = users_collection.find_one({"email": email})
    current_dt = datetime.now(timezone.utc) # MODIFIED: Use datetime

    if user:
        print(f"User {email} already exists.")
        if user.get("status") != STATUS_ACTIVE or user.get("role") != "admin":
            users_collection.update_one(
                {"email": email},
                {"$set": {"status": STATUS_ACTIVE, "role": "admin", "updated_at": current_dt}} # MODIFIED
            )
            print(f"Updated user {email} to ensure admin role and active status.")
        return True
    
    hashed_pass = hash_password(plain_password)
    try:
        users_collection.insert_one({
            "email": email,
            "password": hashed_pass,
            "role": role,
            "status": STATUS_ACTIVE, # Admins are active by default
            "created_at": current_dt, # MODIFIED
            "updated_at": current_dt, # MODIFIED
            "full_name": email.split('@')[0] # Added for consistency
        })
        print(f"Admin user {email} created successfully with active status.")
        return True
    except Exception as e:
        print(f"Error creating admin user {email}: {e}")
        return False

def initialize_all_components(default_db="MongoDB"):
    """Initializes all backend components in the correct order."""
    global gemini_model_genai, together_client, openai_client, embedding_model, faiss_index, BACKEND_INITIAL_LOAD_MSG, mongo_client_instance, mongo_db_obj

    print("--- Backend Initialization Started ---")
    
    print("Step 1: Initializing Database Client...")
    initialize_mongodb_client()

    create_admin_user_if_not_exists("developer@vuega.se", "Vuega-Dev-P@ss!2024")
    
    print("Step 2: Verifying Configurations...")
    if mongo_db_obj is not None:
        backend_get_initial_main_config() # Ensures main config is in DB
        backend_get_current_scraper_payload() # Ensures scraper payload is in DB
    else:
        print("  - WARNING: MongoDB not available, cannot initialize configs in DB.")

    print("Step 3: Initializing AI Service Clients...")
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model_genai = genai.GenerativeModel("gemini-2.5-flash")
            print("  - Google Gemini client configured.")
        except Exception as e:
            print(f"  - Google Gemini client init failed: {e}")
    else:
        print("  - Google Gemini client: SKIPPED (API key not found).")
    
    if TOGETHER_API_KEY:
        try:
            together_client = Together(api_key=TOGETHER_API_KEY)
            print("  - TogetherAI client configured.")
        except Exception as e:
            print(f"  - TogetherAI client init failed: {e}")
    else:
        print("  - TogetherAI client: SKIPPED (API key not found).")

    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("  - OpenAI client configured.")
        except Exception as e:
            print(f"  - OpenAI client init failed: {e}")
    else:
        print("  - OpenAI client: SKIPPED (API key not found).")

    print("Step 4: Initializing Local RAG Components...")
    print("  - Loading SentenceTransformer model 'intfloat/multilingual-e5-base'. This may take a moment...")
    try:
        embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
        print(f"  - SentenceTransformer model loaded (Dimension: {embedding_model.get_sentence_embedding_dimension()}).")
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        print("  - FAISS index shell initialized.")
    except Exception as e:
        print(f"  - CRITICAL ERROR loading SentenceTransformer model: {e}. RAG disabled.")
        BACKEND_INITIAL_LOAD_MSG = "Critical: Embedding model failed. RAG non-functional."
        print("--- Backend Initialization Halted ---")
        return

    print(f"Step 5: Loading initial data from default database: '{default_db}'...")
    BACKEND_INITIAL_LOAD_MSG = load_data_from_selected_db(default_db)
    
    print(f"Final DB Load Status: {BACKEND_INITIAL_LOAD_MSG}")
    print("--- Backend Initialization Complete ---")