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
              {/* <button className="notification-btn">
                <i className="fas fa-bell"></i>
                <span className="notification-badge" id="notificationBadge">0</span>
              </button> */}

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


// import React, { useState, useEffect } from 'react';
// import { Link, useLocation, useNavigate } from 'react-router-dom';
// import { useAuth } from '../context/AuthContext';

// const Navbar = ({ isDashboard = false }) => {
//   const { currentUser, logout } = useAuth();
//   const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
//   const [isUserDropdownOpen, setIsUserDropdownOpen] = useState(false);
//   const [notificationCount, setNotificationCount] = useState(0);
//   const location = useLocation();
//   const navigate = useNavigate();

//   const toggleMobileMenu = () => {
//     setIsMobileMenuOpen(!isMobileMenuOpen);
//   };

//   const closeMobileMenu = () => {
//     setIsMobileMenuOpen(false);
//   };

//   const toggleUserDropdown = () => {
//     setIsUserDropdownOpen(!isUserDropdownOpen);
//   };

//   const closeUserDropdown = () => {
//     setIsUserDropdownOpen(false);
//   };

//   useEffect(() => {
//     const handleClickOutside = (event) => {
//       const dropdown = document.getElementById('userDropdown');
//       if (dropdown && !dropdown.contains(event.target)) {
//         closeUserDropdown();
//       }
//     };

//     document.addEventListener('click', handleClickOutside);
//     return () => document.removeEventListener('click', handleClickOutside);
//   }, []);

//   // WebSocket connection for real-time violation alerts
//   useEffect(() => {
//     if (!isDashboard || !currentUser) {
//       setNotificationCount(0);
//       return;
//     }

//     // Try to extract host from current origin or fallback
//     const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
//     const wsHost = window.location.host;
//     const wsUrl = `${wsProtocol}://${wsHost}/ws`;

//     const ws = new WebSocket(wsUrl);

//     ws.onopen = () => {
//       console.log('Navbar WebSocket connected for violation alerts');
//     };

//     ws.onmessage = (event) => {
//       try {
//         const message = JSON.parse(event.data);

//         // Handle violation broadcast (backend sends violation via queue)
//         if (message.type === 'violation') {
//           const { data } = message;
//           const violationType = (data.violation_type || '').replace('_', ' ').toUpperCase();

//           // Increase notification count
//           setNotificationCount(prev => prev + 1);

//           // Create clickable toast that navigates to dashboard and resets count
//           const toast = document.createElement('div');
//           toast.className = 'toast error clickable-toast';
//           toast.innerHTML = `
//             <strong>VIOLATION ALERT</strong><br/>
//             Stream ${data.stream_id + 1}: ${violationType} (${data.speed_kmh?.toFixed(1) || 'N/A'} km/h)
//           `;

//           // Click handler: navigate to dashboard and reset badge
//           toast.onclick = () => {
//             setNotificationCount(0);
//             navigate('/dashboard');
//             toast.remove();
//           };

//           // Auto-remove after 8 seconds
//           setTimeout(() => {
//             if (toast && toast.parentElement) {
//               toast.remove();
//             }
//           }, 8000);

//           // Add styles if not present
//           if (!document.getElementById('toast-styles')) {
//             const style = document.createElement('style');
//             style.id = 'toast-styles';
//             style.textContent = `
//               .toast {
//                 position: fixed;
//                 bottom: 2rem;
//                 right: 2rem;
//                 padding: 1rem 1.5rem;
//                 background: white;
//                 border-radius: 8px;
//                 box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
//                 z-index: 10000;
//                 animation: slideIn 0.4s ease;
//                 max-width: 320px;
//                 cursor: pointer;
//                 border-left: 5px solid #ef4444;
//                 transition: transform 0.2s;
//               }
//               .toast:hover {
//                 transform: translateY(-4px);
//               }
//               .toast.error {
//                 border-left-color: #ef4444;
//               }
//               .toast.clickable-toast {
//                 cursor: pointer;
//               }
//               .toast strong {
//                 color: #dc2626;
//                 font-size: 1.1em;
//               }
//               @keyframes slideIn {
//                 from {
//                   transform: translateX(400px);
//                   opacity: 0;
//                 }
//                 to {
//                   transform: translateX(0);
//                   opacity: 1;
//                 }
//               }
//             `;
//             document.head.appendChild(style);
//           }

//           document.body.appendChild(toast);
//         }
//       } catch (error) {
//         console.error('Error parsing WebSocket message in Navbar:', error);
//       }
//     };

//     ws.onclose = () => {
//       console.log('Navbar WebSocket disconnected');
//     };

//     ws.onerror = (error) => {
//       console.error('Navbar WebSocket error:', error);
//     };

//     return () => {
//       ws.close();
//     };
//   }, [isDashboard, currentUser, navigate]);

//   // Click on notification bell: go to dashboard and reset count
//   const handleNotificationClick = () => {
//     if (notificationCount > 0) {
//       setNotificationCount(0);
//     }
//     navigate('/dashboard');
//   };

//   const getUserInitials = () => {
//     if (currentUser?.username) {
//       return currentUser.username.substring(0, 2).toUpperCase();
//     }
//     return 'U';
//   };

//   const getDisplayName = () => {
//     return currentUser?.full_name || currentUser?.username || 'User';
//   };

//   const isActive = (path) => location.pathname === path;

//   return (
//     <nav className={`navbar ${isDashboard ? 'navbar-dashboard' : ''}`}>
//       <div className="navbar-container">
//         <div className="navbar-brand">
//           <Link to={isDashboard ? '/dashboard' : '/'}>
//             <img src="/frontend/Logo.png" alt="TMS Logo" className="brand-logo" />
//           </Link>
//         </div>

//         <button
//           className="navbar-toggle"
//           id="mobile-menu-btn"
//           aria-label="Toggle navigation"
//           onClick={toggleMobileMenu}
//         >
//           <i className={`fas ${isMobileMenuOpen ? 'fa-times' : 'fa-bars'}`}></i>
//         </button>

//         <div className={`navbar-menu ${isMobileMenuOpen ? 'active' : ''}`} id="navbar-menu">
//           {isDashboard ? (
//             <>
//               <Link
//                 className={`nav-link ${isActive('/analytics') ? 'active' : ''}`}
//                 to="/analytics"
//                 onClick={closeMobileMenu}
//               >
//                 <i className="fas fa-chart-line"></i> Analytics
//               </Link>
//               <Link
//                 className={`nav-link ${isActive('/monitoring') ? 'active' : ''}`}
//                 to="/monitoring"
//                 onClick={closeMobileMenu}
//               >
//                 <i className="fas fa-video"></i> Monitoring
//               </Link>
//               <Link
//                 className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
//                 to="/dashboard"
//                 onClick={closeMobileMenu}
//               >
//                 <i className="fas fa-gauge"></i> Overview
//               </Link>
//             </>
//           ) : (
//             <>
//               <a className="nav-link active" href="#home" onClick={closeMobileMenu}>
//                 <i className="fas fa-home"></i> Home
//               </a>
//               <a className="nav-link" href="#about" onClick={closeMobileMenu}>
//                 <i className="fas fa-info-circle"></i> About
//               </a>
//               <a className="nav-link" href="#project" onClick={closeMobileMenu}>
//                 <i className="fas fa-project-diagram"></i> Project
//               </a>
//               <a className="nav-link" href="#contact" onClick={closeMobileMenu}>
//                 <i className="fas fa-envelope"></i> Contact
//               </a>
//             </>
//           )}

//           <div className="mobile-auth-btn">
//             {currentUser ? (
//               <>
//                 <div className="mobile-user-info">
//                   <i className="fas fa-user-circle"></i>
//                   <span>{getDisplayName()}</span>
//                 </div>
//                 <button className="btn btn-danger full-width" onClick={logout}>
//                   <i className="fas fa-sign-out-alt"></i> Logout
//                 </button>
//               </>
//             ) : null}
//           </div>
//         </div>

//         <div className="navbar-actions desktop-only">
//           {isDashboard && currentUser ? (
//             <>
//               <button
//                 className="notification-btn"
//                 onClick={handleNotificationClick}
//                 style={{ position: 'relative' }}
//               >
//                 <i className="fas fa-bell"></i>
//                 {notificationCount > 0 && (
//                   <span className="notification-badge" id="notificationBadge">
//                     {notificationCount > 99 ? '99+' : notificationCount}
//                   </span>
//                 )}
//               </button>

//               <div className={`user-dropdown ${isUserDropdownOpen ? 'active' : ''}`} id="userDropdown">
//                 <div className="user-dropdown-toggle" onClick={toggleUserDropdown}>
//                   <div className="user-avatar">{getUserInitials()}</div>
//                   <div className="user-info">
//                     <div className="user-name">{getDisplayName()}</div>
//                     <div className="user-email">{currentUser?.email || ''}</div>
//                   </div>
//                   <i className="fas fa-chevron-down dropdown-arrow"></i>
//                 </div>
//                 <div className="dropdown-menu">
//                   <button className="dropdown-item dropdown-item-danger" onClick={logout}>
//                     <i className="fas fa-sign-out-alt"></i>
//                     <span>Logout</span>
//                   </button>
//                 </div>
//               </div>
//             </>
//           ) : null}
//         </div>
//       </div>
//     </nav>
//   );
// };

// export default Navbar; 