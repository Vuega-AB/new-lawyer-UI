# main_flask_app.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature

# --- Import your backend logic functions ---
from backend import (
    initialize_all_components,
    create_user,
    verify_password,
    get_full_user_for_auth,
    get_user_by_email,
    verify_totp_code,
    STATUS_ACTIVE,
    set_new_password,
    create_user,
    verify_otp_and_activate_user,
    regenerate_otp_for_user
    # Add any other backend functions you need here
)

# ===================================================================
# ||                CREATE EXTENSION OBJECTS                     ||
# ===================================================================
# Create the extension objects here at the global level, but do not
# configure them yet. They are empty shells.
mail = Mail()
ts_password_reset = None # We will create this inside the factory
FRONTEND_EMAIL_TEMPLATE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'frontend', 'pages', 'email')
)
# ===================================================================
# ||                THE APPLICATION FACTORY                        ||
# ===================================================================
def create_app():
    """
    This function creates and configures the Flask application.
    This is the standard, correct pattern.
    """
    app = Flask(__name__)
    CORS(app)

    app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-very-strong-secret-key-for-dev-only")
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() in ['true', '1']
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
    mail.init_app(app)

    global ts_password_reset
    ts_password_reset = URLSafeTimedSerializer(app.config['SECRET_KEY'])

    @app.route("/api/auth/signup", methods=["POST"])
    def api_signup():
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        success, message, otp = create_user(email, password)

        if success:
            # If the user was created, now we send the OTP email
            email_sent = send_system_email(
                to_email=email,
                subject="Your IntelLaw Confirmation Code",
                template_name_no_ext="send_otp", # This matches your file names
                # Pass the OTP as a variable to be replaced in the template
                otp_code=otp
            )
            if not email_sent:
                # You might want to handle cases where the user is created but the email fails
                print(f"WARNING: User {email} was created, but OTP email failed to send.")
            
            return jsonify({"success": True, "message": message}), 201
        else:
            return jsonify({"success": False, "message": message}), 400
        

    @app.route("/api/auth/verify-2fa", methods=["POST"])
    def api_verify_2fa():
        data = request.get_json()
        email = data.get("email")
        submitted_code = data.get("totp_code")

        if not email or not submitted_code:
            return jsonify({"success": False, "message": "Email and 2FA code are required."}), 400

        # Fetch the user from the database to get their secret
        db_user = get_full_user_for_auth(email)
        if not db_user or not db_user.get("totp_secret"):
            return jsonify({"success": False, "message": "2FA is not configured for this user or user not found."}), 404

        # Use your backend function to securely compare the code
        is_valid = verify_totp_code(db_user.get("totp_secret"), submitted_code)

        if is_valid:
            # If the code is correct, the login is fully complete.
            user_info = {"email": db_user.get("email"), "full_name": db_user.get("full_name"), "role": db_user.get("role")}
            return jsonify({
                "success": True,
                "message": "Two-Factor Authentication successful!",
                "user": user_info
            }), 200
        else:
            # If the code is incorrect, send back a failure message.
            return jsonify({"success": False, "message": "Invalid authentication code."}), 401
        
    @app.route("/api/auth/verify-otp", methods=["POST"])
    def api_verify_otp():
        data = request.get_json()
        email = data.get("email")
        otp_code = data.get("otp_code")

        if not email or not otp_code:
            return jsonify({"success": False, "message": "Email and OTP code are required."}), 400

        # Use your existing backend function to check the OTP
        success, message = verify_otp_and_activate_user(email, otp_code)

        if success:
            return jsonify({"success": True, "message": message}), 200
        else:
            # e.g., "Invalid OTP", "OTP has expired"
            return jsonify({"success": False, "message": message}), 400
        
    @app.route("/api/auth/resend-otp", methods=["POST"])
    def api_resend_otp():
        data = request.get_json()
        email = data.get("email")

        if not email:
            return jsonify({"success": False, "message": "Email is required."}), 400

        # Use your existing backend function to generate a new OTP
        new_otp, message = regenerate_otp_for_user(email)

        if new_otp:
            # If a new OTP was generated, trigger the email sending process
            # send_otp_email(email, new_otp) # You would create this helper in email_service.py
            print(f"RESEND OTP: New OTP for {email} is {new_otp}") # For local testing
            return jsonify({"success": True, "message": "A new confirmation code has been sent."}), 200
        else:
            # e.g., "User not found or account already active"
            return jsonify({"success": False, "message": message}), 400
    
    @app.route("/api/auth/reset-password", methods=["POST"])
    def api_reset_password():
        data = request.get_json()
        token = data.get("token")
        new_password = data.get("password")

        if not token or not new_password:
            return jsonify({"success": False, "message": "Token and new password are required."}), 400

        if len(new_password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters long."}), 400

        try:
            # Use the serializer to decode the token and get the email.
            # It will automatically check for expiration (max_age=3600 seconds = 1 hour).
            email = ts_password_reset.loads(token, salt='password-reset-salt', max_age=3600)
        except (SignatureExpired, BadTimeSignature):
            return jsonify({"success": False, "message": "The password reset link is invalid or has expired."}), 400
        except Exception:
            return jsonify({"success": False, "message": "An invalid token was provided."}), 400

        # If the token is valid, use your existing backend function to set the new password.
        success, message = set_new_password(email, new_password)

        if success:
            return jsonify({"success": True, "message": "Your password has been reset successfully. Please log in."}), 200
        else:
            # Pass the message from the backend (e.g., "User not found")
            return jsonify({"success": False, "message": message}), 500
    
    def send_system_email(to_email, subject, template_name_no_ext, **kwargs):
        try:
            html_template_path = os.path.join(FRONTEND_EMAIL_TEMPLATE_PATH, f"{template_name_no_ext}.html")
            text_template_path = os.path.join(FRONTEND_EMAIL_TEMPLATE_PATH, f"{template_name_no_ext}.txt")

            # Read the entire content of the template files
            with open(html_template_path, 'r', encoding='utf-8') as f:
                html_body = f.read()
            with open(text_template_path, 'r', encoding='utf-8') as f:
                text_body = f.read()

            # Perform simple text replacement for any variables passed in kwargs.
            # This replaces placeholders like {{ reset_url }} in your template files.
            for key, value in kwargs.items():
                placeholder = f"{{{{ {key} }}}}" # Creates the string "{{ key }}"
                html_body = html_body.replace(placeholder, str(value))
                text_body = text_body.replace(placeholder, str(value))
            # --- END OF CORRECTED LOGIC ---

            msg = Message(subject, recipients=[to_email], html=html_body, body=text_body)
            mail.send(msg) # This call requires the Flask app context to be active
            print(f"Email '{subject}' sent successfully to {to_email}.")
            return True
        except FileNotFoundError as e:
            print(f"MAIL ERROR: Email template not found. Please check the path: {e}.")
            return False
        except Exception as e:
            print(f"MAIL ERROR: Failed to send email '{subject}' to {to_email}: {e}")
            return False

    @app.route("/api/auth/forgot-password", methods=["POST"])
    def api_forgot_password():
        data = request.get_json()
        email = data.get("email")
        if not email:
            return jsonify({"success": False, "message": "Email is required."}), 400

        user = get_user_by_email(email)
        if user:
            try:
                token = ts_password_reset.dumps(email, salt='password-reset-salt')
                reset_url = f"http://localhost:8001/pages/reset_password_form.html?token={token}"
                
                print(f"PASSWORD RESET LINK for {email}: {reset_url}") # For testing

                # This call now works correctly with the refactored function
                send_system_email(
                    to_email=email,
                    subject="Your IntelLaw Password Reset Link",
                    template_name_no_ext="reset_password_email",
                    # Pass variables to be replaced in the template as keyword arguments
                    reset_url=reset_url,
                    app_name="IntelLaw" 
                )
            except Exception as e:
                print(f"ERROR processing password reset for {email}: {e}")

        # Always return a generic success message for security
        return jsonify({"success": True, "message": "If an account with that email exists, a password reset link has been sent."}), 200


    # main_flask_app.py

    @app.route("/api/auth/login", methods=["POST"])
    def api_login():
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        db_user = get_full_user_for_auth(email)

        if db_user and verify_password(password, db_user.get("password")):
            user_status = db_user.get("status")

            if user_status == STATUS_ACTIVE:
                # This part for active users remains the same
                return jsonify({
                    "success": True, 
                    "message": "Login successful!", 
                    "user": {"email": db_user.get("email"), "full_name": db_user.get("full_name"), "role": db_user.get("role")}
                }), 200
            else:
                # --- THIS IS THE CRITICAL CHANGE ---
                # The password was correct, but the user is not active.
                # We will send back success: false, but ALSO include the specific status.
                return jsonify({
                    "success": False, 
                    "message": f"Account status is '{user_status}'.",
                    "user_status": user_status  # <<< This is the new, clear signal
                }), 403 # 403 Forbidden is a good status code for this
        else:
            # This part for wrong password remains the same
            return jsonify({"success": False, "message": "Invalid credentials."}), 401
    # Return the fully configured app instance
    return app


# ===================================================================
# ||                      START THE SERVER                         ||
# ===================================================================
if __name__ == "__main__":
    app = create_app()
    initialize_all_components()
    
    print("--- Python API Backend is RUNNING and listening on http://127.0.0.1:5000 ---")
    app.run(host="0.0.0.0", port=5000, debug=True)