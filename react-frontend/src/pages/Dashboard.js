import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../context/AuthContext';
import { useConfig } from '../context/ConfigContext';
import { formatTime, formatSpeed } from '../utils/helpers';

const Dashboard = () => {
  const { getAuthHeaders } = useAuth();
  const { API_BASE_URL, POLL_INTERVAL } = useConfig();
  const [stats, setStats] = useState({
    active_streams: 0,
    total_vehicles: 0,
    total_violations: 0,
    violation_summary: {}
  });
  const [violations, setViolations] = useState([]);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [API_BASE_URL, POLL_INTERVAL]);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
        setViolations((data.violations || []).slice(0, 10));
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  return (
    <div className="dashboard-body">
      <Navbar isDashboard={true} />

      <div className="dashboard-main">
        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-heart-pulse"></i>
              Current Status
            </h2>
          </div>
          <div className="stats-grid">
            <div className="stat-card">
              <i className="fas fa-video stat-card-icon"></i>
              <div className="stat-card-label">Online Streams</div>
              <div className="stat-card-value">{stats.active_streams || 0}</div>
              <div className="stat-card-unit">Processing Live</div>
            </div>
            <div className="stat-card">
              <i className="fas fa-car-on stat-card-icon"></i>
              <div className="stat-card-label">Today's Traffic</div>
              <div className="stat-card-value">{stats.total_vehicles || 0}</div>
              <div className="stat-card-unit">Vehicles Detected</div>
            </div>
            <div className="stat-card">
              <i className="fas fa-triangle-exclamation stat-card-icon"></i>
              <div className="stat-card-label">Today's Violations</div>
              <div className="stat-card-value">{stats.total_violations || 0}</div>
              <div className="stat-card-unit">Alerts Generated</div>
            </div>
            <div className="stat-card">
              <i className="fas fa-gauge-simple-high stat-card-icon"></i>
              <div className="stat-card-label">Critical Speeding</div>
              <div className="stat-card-value">{stats.violation_summary?.speed || 0}</div>
              <div className="stat-card-unit">Major Incidents</div>
            </div>
          </div>
        </section>

        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-satellite-dish"></i>
              Live Alert Feed
            </h2>
          </div>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Source Stream</th>
                  <th>Alert Type</th>
                  <th>Detected Speed</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {violations.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="table-empty-state">
                      <i className="fas fa-inbox"></i>
                      <p>No violations</p>
                    </td>
                  </tr>
                ) : (
                  violations.map((v, index) => (
                    <tr key={index}>
                      <td>Stream {v.stream_id}</td>
                      <td>{(v.violation_type || '').replace('_', ' ').toUpperCase()}</td>
                      <td>{formatSpeed(v.speed_kmh)}</td>
                      <td>{formatTime(v.timestamp)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
};

export default Dashboard;
