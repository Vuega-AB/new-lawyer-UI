
const currentUser = localStorage.getItem('currentUserEmail');
if (!currentUser) {
    window.location.href = 'auth/pages/login.html';
} else {
    console.log(`Welcome, ${currentUser}! Loading main application...`);
}

document.addEventListener('DOMContentLoaded', () => {
    const logoutButton = document.getElementById('logout-btn'); // Add a button with this ID
    if (logoutButton) {
        logoutButton.addEventListener('click', () => {
            localStorage.removeItem('currentUserEmail');
            window.location.href = 'auth/pages/login.html';
        });
    }
});