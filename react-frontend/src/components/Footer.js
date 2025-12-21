import React from 'react';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-section">
          <h4 className="footer-heading">
            <i className="fas fa-traffic-light"></i> TMS System
          </h4>
          <p className="footer-text">
            Advanced AI-powered traffic monitoring for safer cities.
          </p>
        </div>
        <div className="footer-section">
          <h4 className="footer-heading">
            <i className="fas fa-cogs"></i> Technology
          </h4>
          <p className="footer-text">✓ YOLOv8 Vehicle Detection</p>
          <p className="footer-text">✓ Real-time Processing</p>
        </div>
        <div className="footer-section">
          <h4 className="footer-heading">
            <i className="fas fa-phone"></i> Support
          </h4>
          <p className="footer-text">
            Email: <a href="mailto:support@tms.com">support@tms.com</a>
          </p>
          <p className="footer-text">
            Status: <a href="/dashboard">System Online</a>
          </p>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; 2024 TMS - Traffic Monitoring System. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;
