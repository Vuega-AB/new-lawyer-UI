// js/auth.js

// The local address of your RUNNING Python script's "channel"
const API_BASE_URL = 'http://127.0.0.1:5000';
console.log("rerer???????????????");
// js/auth.js

// js/auth.js

async function handleLogin(form) {
    console.log("handleLogin function was called!");
    const email = form.querySelector('#email').value;
    const password = form.querySelector('#password').value;
    const messageContainer = document.getElementById('message-container');
    messageContainer.textContent = '';

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        const result = await response.json();

        if (response.ok && result.success) {
            // SUCCESS! Now, check the user's role to decide where to go next.
            const user = result.user;

            // =======================================================
            // ||                THIS IS THE NEW LOGIC              ||
            // =======================================================
            if (user.role === 'admin') {
                // ---- ADMIN USER FLOW ----
                console.log("Admin user detected. Bypassing 2FA and logging in directly.");
                alert("Admin login successful!");

                // Set the final 'currentUser' key. The main app's auth guard will see this.
                localStorage.setItem('currentUser', JSON.stringify(user));

                // Redirect directly to the main application.
                window.location.href = '##########################';
            } else {
                // ---- REGULAR USER FLOW ----
                console.log("Regular user detected. Proceeding to 2FA verification.");

                // Set the temporary 'user_pending_2fa' key for the 2FA page to use.
                localStorage.setItem('user_pending_2fa', JSON.stringify(user));

                // Redirect to the 2FA page.
                window.location.href = 'login_2fa.html'; // Both login.html and login_2fa.html are in the 'pages' folder.
            }
            // =======================================================

        } else {
            // Handle login failure from the server
            messageContainer.textContent = result.message || 'Login failed. Please check your credentials.';
        }

    } catch (error) {
        // Handle network errors
        console.error("A network error occurred:", error);
        messageContainer.textContent = 'Network Error: Could not connect to the server.';
    }
}

// Attach the handleLogin function to the form's submit button
document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            handleLogin(loginForm);
        });
    }
});