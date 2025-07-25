# main_flask_app.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Explicitly import all the functions you need from your backend ---
# This is much safer and clearer than a wildcard import.
from backend import (
    initialize_all_components,
    create_user,
    verify_password,
    get_full_user_for_auth,
    retrieve_context_from_db,
    generate_response_together_ai, # Import your different AI generators
    generate_response_gemini,
    generate_response_openai_api,
    process_and_add_pdf_core,
    delete_files_backend,
    get_unique_filenames_from_text_store,
    STATUS_ACTIVE,
    AVAILABLE_MODELS_DICT,
    verify_totp_code,
)

# --- Create the Flask App ---
app = Flask(__name__)
# This is CRITICAL. It allows your HTML/JS frontend to make requests to this server.
CORS(app)


# ===================================================================
# ||                AUTHENTICATION API ENDPOINTS                   ||
# ===================================================================
@app.route("/api/auth/verify-2fa", methods=["POST"])
def api_verify_2fa():
    data = request.get_json()
    email = data.get("email")
    submitted_code = data.get("totp_code")

    if not email or not submitted_code:
        return jsonify({"success": False, "message": "Email and 2FA code are required."}), 400

    # 1. Fetch the user from the database to get their secret
    db_user = get_full_user_for_auth(email)
    if not db_user or not db_user.get("totp_secret"):
        # This is a security measure. If a user doesn't have 2FA, this endpoint should fail.
        return jsonify({"success": False, "message": "2FA is not configured for this user or user not found."}), 404

    # 2. Use your existing backend function to perform the secure comparison
    is_valid = verify_totp_code(db_user.get("totp_secret"), submitted_code)

    if is_valid:
        # 3. If the code is correct, the login is fully complete.
        # Send back the final user object for the frontend to save.
        user_info = {"email": db_user.get("email"), "full_name": db_user.get("full_name")}
        return jsonify({
            "success": True,
            "message": "Two-Factor Authentication successful!",
            "user": user_info
        }), 200
    else:
        # 4. If the code is incorrect, send back a failure message.
        return jsonify({"success": False, "message": "Invalid authentication code."}), 401
    
@app.route("/api/auth/signup", methods=["POST"])
def api_signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    success, message, otp = create_user(email, password)
    if success:
        # Here you could trigger your email sending function if you set it up
        return jsonify({"success": True, "message": message}), 201
    else:
        return jsonify({"success": False, "message": message}), 400

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    db_user = get_full_user_for_auth(email)
    if db_user and verify_password(password, db_user.get("password")):
        if db_user.get("status") == STATUS_ACTIVE:
            return jsonify({
                "success": True,
                "message": "Login successful!",
                "user": {
                    "email": db_user.get("email"),
                    "full_name": db_user.get("full_name"),
                    "role": db_user.get("role") # <<< THIS IS THE CRITICAL PIECE OF DATA
                }
            }), 200
        else:
            return jsonify({"success": False, "message": f"Account status is '{db_user.get('status')}'."}), 403
    else:
        return jsonify({"success": False, "message": "Invalid credentials."}), 401


# ===================================================================
# ||                    CORE APP API ENDPOINTS                     ||
# ===================================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_input = data.get("query")
    model_id = data.get("model_id", "meta-llama/Llama-3-8B-Instruct-hf") # Default model

    if not user_input:
        return jsonify({"success": False, "error": "Query cannot be empty."}), 400

    # 1. Get context from your RAG system
    context_text = retrieve_context_from_db(user_input, top_k=5)

    # 2. Generate a response using the selected AI model
    model_detail = AVAILABLE_MODELS_DICT.get(model_id, {})
    model_type = model_detail.get("type")
    
    response_content = "Error: Model not found or not configured."
    if model_type == "together":
        response_content = generate_response_together_ai(user_input, context_text, model_id, 0.7, 0.9, "You are a helpful assistant.")
    elif model_type == "gemini":
        response_content = generate_response_gemini(user_input, context_text, 0.7, 0.9, "You are a helpful assistant.")
    # Add other model types as needed...

    return jsonify({"success": True, "response": response_content, "context": context_text})


@app.route("/api/files/upload", methods=["POST"])
def api_upload_pdf():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file."}), 400

    if file and file.filename.endswith('.pdf'):
        pdf_bytes = file.read()
        file_name = file.filename
        # Use your existing PDF processing function from backend.py
        message, success = process_and_add_pdf_core(pdf_bytes, file_name, "MongoDB")
        if success:
            return jsonify({"success": True, "message": message, "filenames": get_unique_filenames_from_text_store()})
        else:
            return jsonify({"success": False, "message": message}), 500

    return jsonify({"success": False, "message": "Invalid file type. Only PDF is allowed."}), 400


# ===================================================================
# ||                      START THE SERVER                         ||
# ===================================================================

if __name__ == "__main__":
    # This initializes your MongoDB connection, AI models, etc., when the server starts.
    initialize_all_components(default_db="MongoDB")
    
    print("--- Python API Backend is RUNNING and listening on http://127.0.0.1:5000 ---")
    # This starts the web server.
    app.run(host="0.0.0.0", port=5000, debug=True)