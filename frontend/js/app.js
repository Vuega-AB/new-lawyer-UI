// js/app.js

// --- AUTH GUARD ---
// This runs immediately to protect the page.
const currentUser = localStorage.getItem('currentUserEmail');
if (!currentUser) {
    // If no one is logged in, kick them back to the login page.
    window.location.href = 'pages/login.html';
} else {
    // If they are logged in, we can load the app.
    console.log(`Welcome, ${currentUser}! Loading main application...`);
    // You will add your dashboard/timeline rendering logic here.
}


// --- LOGOUT LOGIC ---
document.addEventListener('DOMContentLoaded', () => {
    const logoutButton = document.getElementById('logout-btn'); // Add a button with this ID
    if (logoutButton) {
        logoutButton.addEventListener('click', () => {
            localStorage.removeItem('currentUserEmail');
            window.location.href = 'pages/login.html';
        });
    }
});