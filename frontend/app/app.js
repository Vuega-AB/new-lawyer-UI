// document.addEventListener('DOMContentLoaded', function () {
//     const mainContent = document.getElementById('main-content');
//     const navLinks = document.querySelectorAll('.nav-link');

//     // Function to load and display page content
//     async function loadPage(pageName) {
//         try {
//             const response = await fetch(`pages/${pageName}.html`);
//             if (!response.ok) throw new Error(`Page not found: ${pageName}.html`);
            
//             const html = await response.text();
            
//             // Use a temporary div to parse the HTML string
//             const tempDiv = document.createElement('div');
//             tempDiv.innerHTML = html;

//             // Clear previous content
//             mainContent.innerHTML = '';
            
//             // Append the new content's children (HTML, STYLE) to the main area
//             // This ensures styles are applied.
//             mainContent.append(...tempDiv.children);

//             // Find and execute script tags manually
//             // This is the key part for making the component's JS run.
//             const scripts = mainContent.querySelectorAll('script');
//             scripts.forEach(script => {
//                 const newScript = document.createElement('script');
//                 // Copy the content of the original script to the new one
//                 newScript.textContent = script.textContent;
//                 // Append the new script to the body to execute it
//                 document.body.appendChild(newScript).remove();
//             });

//         } catch (error) {
//             console.error('Error loading page:', error);
//             mainContent.innerHTML = `<div class="card"><p>Sorry, an error occurred.</p></div>`;
//         }
//     }

//     // --- Event Listeners and Initial Load (no changes here) ---

//     navLinks.forEach(link => {
//         link.addEventListener('click', function (event) {
//             event.preventDefault();
//             const pageId = this.dataset.page;
//             loadPage(pageId);
//             navLinks.forEach(navLink => navLink.classList.remove('active-link'));
//             this.classList.add('active-link');
//         });
//     });

//     const defaultActiveLink = document.querySelector('.nav-link.active-link');
//     if (defaultActiveLink) {
//         loadPage(defaultActiveLink.dataset.page);
//     }
// });