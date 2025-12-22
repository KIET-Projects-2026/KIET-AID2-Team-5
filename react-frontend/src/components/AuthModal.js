import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { showToast } from '../utils/helpers';

const AuthModal = ({ isOpen, onClose, initialTab = 'login' }) => {
  const [activeTab, setActiveTab] = useState(initialTab);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, signup } = useAuth();
  const navigate = useNavigate();

  // Login form state
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');

  // Signup form state
  const [signupName, setSignupName] = useState('');
  const [signupUsername, setSignupUsername] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPassword, setSignupPassword] = useState('');

  React.useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  React.useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  const clearError = () => setError('');

  const handleLogin = async (e) => {
    e.preventDefault();
    clearError();
    setLoading(true);

    const result = await login(loginEmail, loginPassword);
    setLoading(false);

    if (result.success) {
      onClose();
      showToast(`Welcome back, ${result.user.username}!`, 'success');
      navigate('/monitoring');
    } else {
      setError(result.error);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    clearError();
    setLoading(true);

    const result = await signup(signupUsername, signupEmail, signupPassword, signupName);
    setLoading(false);

    if (result.success) {
      onClose();
      showToast(`Welcome to TMS, ${result.user.username}!`, 'success');
      navigate('/monitoring');
    } else {
      setError(result.error);
    }
  };

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) onClose();
  };

  if (!isOpen) return null;

  return (
    <div className={`auth-overlay ${isOpen ? 'active' : ''}`} onClick={handleOverlayClick}>
      <div className="auth-container">
        <div className="auth-header">
          <div className="auth-logo">TMS</div>
          <h2 className="auth-title">Welcome to TMS</h2>
          <p className="auth-subtitle">Sign up or login to access the monitoring dashboard</p>
        </div>

        <div className="auth-tabs">
          <button
            className={`auth-tab ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('login');
              clearError();
            }}
          >
            Login
          </button>
          <button
            className={`auth-tab ${activeTab === 'signup' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('signup');
              clearError();
            }}
          >
            Sign Up
          </button>
        </div>

        {error && (
          <div className="auth-error show">
            <i className="fas fa-exclamation-circle"></i>
            <span>{error}</span>
          </div>
        )}

        {activeTab === 'login' ? (
          <form className="auth-form active" onSubmit={handleLogin}>
            <div className="form-group">
              <label className="form-label">Email Address</label>
              <div className="form-input-icon">
                <i className="fas fa-envelope"></i>
                <input
                  type="email"
                  className="form-input"
                  placeholder="Enter your email"
                  required
                  value={loginEmail}
                  onChange={(e) => setLoginEmail(e.target.value)}
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Password</label>
              <div className="form-input-icon">
                <i className="fas fa-lock"></i>
                <input
                  type="password"
                  className="form-input"
                  placeholder="Enter your password"
                  required
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                />
              </div>
            </div>
            <button type="submit" className="auth-btn" disabled={loading}>
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Please wait...</span>
                </>
              ) : (
                <span>Login</span>
              )}
            </button>
          </form>
        ) : (
          <form className="auth-form active" onSubmit={handleSignup}>
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <div className="form-input-icon">
                <i className="fas fa-user"></i>
                <input
                  type="text"
                  className="form-input"
                  placeholder="Enter your full name"
                  required
                  value={signupName}
                  onChange={(e) => setSignupName(e.target.value)}
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Username</label>
              <div className="form-input-icon">
                <i className="fas fa-at"></i>
                <input
                  type="text"
                  className="form-input"
                  placeholder="Choose a username"
                  required
                  minLength="3"
                  value={signupUsername}
                  onChange={(e) => setSignupUsername(e.target.value)}
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Email Address</label>
              <div className="form-input-icon">
                <i className="fas fa-envelope"></i>
                <input
                  type="email"
                  className="form-input"
                  placeholder="Enter your email"
                  required
                  value={signupEmail}
                  onChange={(e) => setSignupEmail(e.target.value)}
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Password</label>
              <div className="form-input-icon">
                <i className="fas fa-lock"></i>
                <input
                  type="password"
                  className="form-input"
                  placeholder="Create a password (min 6 chars)"
                  required
                  minLength="6"
                  value={signupPassword}
                  onChange={(e) => setSignupPassword(e.target.value)}
                />
              </div>
            </div>
            <button type="submit" className="auth-btn" disabled={loading}>
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Please wait...</span>
                </>
              ) : (
                <span>Create Account</span>
              )}
            </button>
          </form>
        )}

        <div className="auth-divider">
          <span>Secure Access</span>
        </div>

        <p className="auth-security-note">
          <i className="fas fa-shield-alt"></i>
          Your account protects access to the live monitoring dashboard and violation records.
        </p>
      </div>
    </div>
  );
};

export default AuthModal;
