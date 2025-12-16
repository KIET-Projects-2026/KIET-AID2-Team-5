
// ==================== CONFIGURATION ====================
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : 'https://traffic-monitoring-api.onrender.com'; // keep in sync with dashboard

let authToken = null;
let currentUser = null;

// ==================== AUTH MODAL HELPERS ====================
function showAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.classList.add('active');
    const emailInput = document.getElementById('loginEmail');
    if (emailInput) emailInput.focus();
}

function hideAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.classList.remove('active');
    clearAuthError();
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    if (loginForm) loginForm.reset();
    if (signupForm) signupForm.reset();
}

function switchAuthTab(tab) {
    const tabs = document.querySelectorAll('.auth-tab');
    const forms = document.querySelectorAll('.auth-form');

    tabs.forEach(t => t.classList.remove('active'));
    forms.forEach(f => f.classList.remove('active'));

    if (tab === 'login') {
        if (tabs[0]) tabs[0].classList.add('active');
        const form = document.getElementById('loginForm');
        if (form) form.classList.add('active');
    } else {
        if (tabs[1]) tabs[1].classList.add('active');
        const form = document.getElementById('signupForm');
        if (form) form.classList.add('active');
    }
    clearAuthError();
}

function openSignup() {
    switchAuthTab('signup');
    showAuthModal();
}

function openLogin() {
    switchAuthTab('login');
    showAuthModal();
}

function showAuthError(message) {
    const errorEl = document.getElementById('authError');
    const textEl = document.getElementById('authErrorText');
    if (!errorEl || !textEl) return;
    textEl.textContent = message;
    errorEl.classList.add('show');
}

function clearAuthError() {
    const errorEl = document.getElementById('authError');
    if (!errorEl) return;
    errorEl.classList.remove('show');
}

function setButtonLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    if (loading) {
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div><span>Please wait...</span>';
    } else {
        btn.disabled = false;
        btn.innerHTML = '<span>' + (btnId.includes('login') ? 'Login' : 'Create Account') + '</span>';
    }
}

// ==================== AUTH HANDLERS ====================
async function handleLogin(event) {
    event.preventDefault();
    clearAuthError();
    setButtonLoading('loginSubmitBtn', true);

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Login failed');
        }

        authToken = data.access_token;
        currentUser = data.user;
        localStorage.setItem('authToken', authToken);
        localStorage.setItem('currentUser', JSON.stringify(currentUser));

        hideAuthModal();
        showToast(`Welcome back, ${currentUser.username}!`, 'success');

        // After successful login, send user to dashboard
        window.location.href = '/dashboard';
    } catch (error) {
        showAuthError(error.message);
    } finally {
        setButtonLoading('loginSubmitBtn', false);
    }
}

async function handleSignup(event) {
    event.preventDefault();
    clearAuthError();
    setButtonLoading('signupSubmitBtn', true);

    const full_name = document.getElementById('signupName').value;
    const username = document.getElementById('signupUsername').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password, full_name })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Signup failed');
        }

        authToken = data.access_token;
        currentUser = data.user;
        localStorage.setItem('authToken', authToken);
        localStorage.setItem('currentUser', JSON.stringify(currentUser));

        hideAuthModal();
        showToast(`Welcome to TMS, ${currentUser.username}!`, 'success');

        // After successful signup, send user to dashboard
        window.location.href = '/dashboard';
    } catch (error) {
        showAuthError(error.message);
    } finally {
        setButtonLoading('signupSubmitBtn', false);
    }
}

// ==================== UTILITIES ====================
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// Close auth modal when clicking outside
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('authModal');
    if (modal) {
        modal.addEventListener('click', function (event) {
            if (event.target === this) {
                hideAuthModal();
            }
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            hideAuthModal();
        }
    });

    // If already authenticated, offer a quick redirect
    const savedToken = localStorage.getItem('authToken');
    const savedUser = localStorage.getItem('currentUser');
    if (savedToken && savedUser) {
        authToken = savedToken;
        currentUser = JSON.parse(savedUser);
        // Optional: verify token, but even without it we just redirect to dashboard,
        // where token verification is already handled.
    }
});


