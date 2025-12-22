import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';
import { Line, Bar, Pie } from 'react-chartjs-2';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../context/AuthContext';
import { useConfig } from '../context/ConfigContext';
import { formatTime, formatSpeed } from '../utils/helpers';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend);

const Analytics = () => {
  const { getAuthHeaders } = useAuth();
  const { API_BASE_URL } = useConfig();
  const [dateRange, setDateRange] = useState('all');
  const [customDate, setCustomDate] = useState('');
  const [violations, setViolations] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 5;

  const [summary, setSummary] = useState({
    total: 0,
    speed: 0,
    red_light: 0,
    stop_line: 0,
    lane_change: 0
  });

  useEffect(() => {
    fetchViolations();
  }, [dateRange, customDate]);

  const fetchViolations = async () => {
    let query = '?limit=500';
    
    if (dateRange === 'custom' && customDate) {
      query += `&specific_date=${encodeURIComponent(customDate)}`;
    } else if (dateRange !== 'all') {
      query += `&date_range=${encodeURIComponent(dateRange)}`;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/db/violations${query}`, {
        headers: getAuthHeaders()
      });
      
      if (response.ok) {
        const data = await response.json();
        const violationsList = data.violations || [];
        setViolations(violationsList);
        setCurrentPage(1);

        // Calculate summary
        const counts = {
          total: violationsList.length,
          speed: 0,
          red_light: 0,
          stop_line: 0,
          lane_change: 0,
          unsafe_distance: 0
        };

        violationsList.forEach(v => {
          const type = v.violation_type;
          if (counts.hasOwnProperty(type)) {
            counts[type]++;
          }
        });

        setSummary(counts);
      }
    } catch (error) {
      console.error('Error fetching violations:', error);
    }
  };

  // Pagination
  const totalPages = Math.ceil(violations.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const currentViolations = violations.slice(startIndex, endIndex);

  const handlePageChange = (page) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  // Chart data
  const violationsByTypeData = {
    labels: ['Speed', 'Red Light', 'Stop Line', 'Lane Change'],
    datasets: [{
      label: 'Violations',
      data: [summary.speed, summary.red_light, summary.stop_line, summary.lane_change],
      backgroundColor: 'rgba(6, 182, 212, 0.6)',
      borderColor: 'rgba(6, 182, 212, 1)',
      borderWidth: 1
    }]
  };

  const violationsByStreamData = () => {
    const streamCounts = {};
    violations.forEach(v => {
      const key = `Stream ${v.stream_id + 1}`;      streamCounts[key] = (streamCounts[key] || 0) + 1;
    });

    return {
      labels: Object.keys(streamCounts),
      datasets: [{
        data: Object.values(streamCounts),
        backgroundColor: ['#f87171', '#fb923c', '#fbbf24', '#a3e635']
      }]
    };
  };

  const violationsOverTimeData = () => {
    const buckets = {};
    violations.forEach(v => {
      if (!v.timestamp) return;
      const d = new Date(v.timestamp);
      if (isNaN(d.getTime())) return;
      const key = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      buckets[key] = (buckets[key] || 0) + 1;
    });

    const sortedLabels = Object.keys(buckets).sort();

    return {
      labels: sortedLabels,
      datasets: [{
        label: 'Violations',
        data: sortedLabels.map(k => buckets[k]),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        fill: true,
        tension: 0.3
      }]
    };
  };

  const speedDistributionData = () => {
    const speedBuckets = [0, 0, 0, 0]; // 0-30, 30-60, 60-90, 90+
    violations.forEach(v => {
      const speed = v.speed_kmh || 0;
      if (speed < 30) speedBuckets[0]++;
      else if (speed < 60) speedBuckets[1]++;
      else if (speed < 90) speedBuckets[2]++;
      else speedBuckets[3]++;
    });

    return {
      labels: ['0-30', '30-60', '60-90', '90+'],
      datasets: [{
        label: 'Vehicle Count',
        data: speedBuckets,
        backgroundColor: 'rgba(139, 92, 246, 0.6)'
      }]
    };
  };

  return (
    <div className="dashboard-body">
      <Navbar isDashboard={true} />

      <div className="dashboard-main">
        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-filter"></i>
              Report Configuration
            </h2>
          </div>
          <div className="filter-container">
            <select
              className="form-select"
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
            >
              <option value="all">Lifetime Data</option>
              <option value="today">Today's Report</option>
              <option value="yesterday">Yesterday</option>
              <option value="last_week">Past 7 Days</option>
              <option value="last_month">Past 30 Days</option>
              <option value="last_year">Past Year</option>
              <option value="custom">Select Custom Range...</option>
            </select>
            {dateRange === 'custom' && (
              <input
                type="date"
                className="form-input"
                value={customDate}
                onChange={(e) => setCustomDate(e.target.value)}
              />
            )}
          </div>
        </section>

        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-calculator"></i>
              Aggregated Metrics
            </h2>
          </div>
          <div className="stats-grid">
            <div className="stat-card">
              <i className="fas fa-folder-open stat-card-icon"></i>
              <div className="stat-card-label">Total Incidents</div>
              <div className="stat-card-value">{summary.total}</div>
              <div className="stat-card-unit">Recorded in period</div>
            </div>
            <div className="stat-card">
              <i className="fas fa-gauge-high stat-card-icon"></i>
              <div className="stat-card-label">Speeding Events</div>
              <div className="stat-card-value">{summary.speed}</div>
              <div className="stat-card-unit">Recorded in period</div>
            </div>
            <div className="stat-card">
              <i className="fas fa-traffic-light stat-card-icon"></i>
              <div className="stat-card-label">Red Light Violations</div>
              <div className="stat-card-value">{summary.red_light}</div>
              <div className="stat-card-unit">Recorded in period</div>
            </div>
          </div>
        </section>

        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-chart-area"></i>
              Deep Dive Analytics
            </h2>
          </div>
          <div className="chart-grid">
            <div className="chart-card">
              <div className="chart-card-header">
                <h3 className="chart-card-title">Violation Trends Over Time</h3>
              </div>
              <Line data={violationsOverTimeData()} options={{ responsive: true, maintainAspectRatio: true }} />
            </div>
            
            <div className="chart-card">
              <div className="chart-card-header">
                <h3 className="chart-card-title">Violation Type Breakdown</h3>
              </div>
              <Bar data={violationsByTypeData} options={{ responsive: true, maintainAspectRatio: true }} />
            </div>

            <div className="chart-card">
              <div className="chart-card-header">
                <h3 className="chart-card-title">Violations by Camera Stream</h3>
              </div>
              <Pie data={violationsByStreamData()} options={{ responsive: true, maintainAspectRatio: true }} />
            </div>

            <div className="chart-card">
              <div className="chart-card-header">
                <h3 className="chart-card-title">Vehicle Speed Distribution</h3>
              </div>
              <Bar data={speedDistributionData()} options={{ responsive: true, maintainAspectRatio: true }} />
            </div>
          </div>
        </section>

        <section className="content-section">
          <div className="section-header">
            <h2 className="section-title">
              <i className="fas fa-table-list"></i>
              Detailed Incident Log
            </h2>
          </div>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Source Stream</th>
                  <th>Violation Category</th>
                  <th>Detected Speed</th>
                  <th>Signal Status</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {currentViolations.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="table-empty-state">
                      <i className="fas fa-inbox"></i>
                      <p>No violations found for selected period</p>
                    </td>
                  </tr>
                ) : (
                  currentViolations.map((v, index) => (
                    <tr key={index}>
                      <td>Stream {v.stream_id + 1}</td>

                      <td>{(v.violation_type || '').replace('_', ' ').toUpperCase()}</td>
                      <td>{formatSpeed(v.speed_kmh)}</td>
                      <td>{v.signal_state || 'N/A'}</td>
                      <td>{formatTime(v.timestamp)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
            {violations.length > 0 && (
              <div className="table-footer">
                <div className="pagination-info">
                  Showing {startIndex + 1}-{Math.min(endIndex, violations.length)} of {violations.length}
                </div>
                <div className="pagination-controls">
                  <button
                    className="page-btn"
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                  >
                    <i className="fas fa-chevron-left"></i>
                  </button>
                  {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1).map(page => (
                    <button
                      key={page}
                      className={`page-btn ${page === currentPage ? 'active' : ''}`}
                      onClick={() => handlePageChange(page)}
                    >
                      {page}
                    </button>
                  ))}
                  <button
                    className="page-btn"
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                  >
                    <i className="fas fa-chevron-right"></i>
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
};

export default Analytics;
