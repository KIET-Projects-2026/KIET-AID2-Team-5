import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Navbar = ({ isDashboard = false }) => {
  const { currentUser, logout } = useAuth();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isUserDropdownOpen, setIsUserDropdownOpen] = useState(false);
  const location = useLocation();

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  const toggleUserDropdown = () => {
    setIsUserDropdownOpen(!isUserDropdownOpen);
  };

  const closeUserDropdown = () => {
    setIsUserDropdownOpen(false);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      const dropdown = document.getElementById('userDropdown');
      if (dropdown && !dropdown.contains(event.target)) {
        closeUserDropdown();
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const getUserInitials = () => {
    if (currentUser?.username) {
      return currentUser.username.substring(0, 2).toUpperCase();
    }
    return 'U';
  };

  const getDisplayName = () => {
    return currentUser?.full_name || currentUser?.username || 'User';
  };

  const isActive = (path) => location.pathname === path;

  return (
    <nav className={`navbar ${isDashboard ? 'navbar-dashboard' : ''}`}>
      <div className="navbar-container">
        <div className="navbar-brand">
          <img src="/Logo.png" alt="TMS Logo" className="brand-logo" />
        </div>

        <button
          className="navbar-toggle"
          id="mobile-menu-btn"
          aria-label="Toggle navigation"
          onClick={toggleMobileMenu}
        >
          <i className={`fas ${isMobileMenuOpen ? 'fa-times' : 'fa-bars'}`}></i>
        </button>

        <div className={`navbar-menu ${isMobileMenuOpen ? 'active' : ''}`} id="navbar-menu">
          {isDashboard ? (
            <>
              <Link
                className={`nav-link ${isActive('/analytics') ? 'active' : ''}`}
                to="/analytics"
                onClick={closeMobileMenu}
              >
                <i className="fas fa-chart-line"></i> Analytics
              </Link>
              <Link
                className={`nav-link ${isActive('/monitoring') ? 'active' : ''}`}
                to="/monitoring"
                onClick={closeMobileMenu}
              >
                <i className="fas fa-video"></i> Monitoring
              </Link>
              <Link
                className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
                to="/dashboard"
                onClick={closeMobileMenu}
              >
                <i className="fas fa-gauge"></i> Overview
              </Link>
            </>
          ) : (
            <>
              <a className="nav-link active" href="#home" onClick={closeMobileMenu}>
                <i className="fas fa-home"></i> Home
              </a>
              <a className="nav-link" href="#about" onClick={closeMobileMenu}>
                <i className="fas fa-info-circle"></i> About
              </a>
              <a className="nav-link" href="#project" onClick={closeMobileMenu}>
                <i className="fas fa-project-diagram"></i> Project
              </a>
              <a className="nav-link" href="#contact" onClick={closeMobileMenu}>
                <i className="fas fa-envelope"></i> Contact
              </a>
            </>
          )}

          <div className="mobile-auth-btn">
            {currentUser ? (
              <>
                <div className="mobile-user-info">
                  <i className="fas fa-user-circle"></i>
                  <span>{getDisplayName()}</span>
                </div>
                <button className="btn btn-danger full-width" onClick={logout}>
                  <i className="fas fa-sign-out-alt"></i> Logout
                </button>
              </>
            ) : null}
          </div>
        </div>

        <div className="navbar-actions desktop-only">
          {isDashboard && currentUser ? (
            <>
              <button className="notification-btn">
                <i className="fas fa-bell"></i>
                <span className="notification-badge" id="notificationBadge">0</span>
              </button>

              <div className={`user-dropdown ${isUserDropdownOpen ? 'active' : ''}`} id="userDropdown">
                <div className="user-dropdown-toggle" onClick={toggleUserDropdown}>
                  <div className="user-avatar">{getUserInitials()}</div>
                  <div className="user-info">
                    <div className="user-name">{getDisplayName()}</div>
                    <div className="user-email">{currentUser?.email || ''}</div>
                  </div>
                  <i className="fas fa-chevron-down dropdown-arrow"></i>
                </div>
                <div className="dropdown-menu">
                  <button className="dropdown-item dropdown-item-danger" onClick={logout}>
                    <i className="fas fa-sign-out-alt"></i>
                    <span>Logout</span>
                  </button>
                </div>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
