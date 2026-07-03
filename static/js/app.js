// Frontend interactions & enhancements
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('loanForm');
    
    if (form) {
        form.addEventListener('submit', (e) => {
            const submitBtn = form.querySelector('.btn-submit');
            submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch animate-spin"></i> Processing Evaluation...';
            submitBtn.style.opacity = '0.7';
            submitBtn.style.pointerEvents = 'none';
        });
    }
});

// Simple spinner animation style injected dynamically if not already in CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .animate-spin {
        animation: spin 1s linear infinite;
        display: inline-block;
    }
`;
document.head.appendChild(style);
