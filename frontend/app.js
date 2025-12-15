// API Configuration - Auto-detect environment
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://trafficmonitoringsystem.onrender.com';

const WS_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'ws://localhost:8000'
    : 'wss://trafficmonitoringsystem.onrender.com';

console.log('Environment detected:', window.location.hostname);
console.log('API Base:', API_BASE);
console.log('WebSocket Base:', WS_BASE);

let ws = null;
let allViolations = [];
let streamStates = [false, false, false, false];
let streamIntervals = [null, null, null, null];
let streamRetryCount = [0, 0, 0, 0];
const MAX_RETRY_COUNT = 10;

// ==================== STREAM RENDERING ====================

function startStreamRendering(streamId) {
    const imgElement = document.getElementById(`stream${streamId}`);
    const placeholder = document.getElementById(`placeholder${streamId}`);
    
    if (!imgElement) {
        console.error(`Stream ${streamId}: Image element not found`);
        return;
    }
    
    // Reset retry count when starting fresh
    streamRetryCount[streamId] = 0;
    
    console.log(`Starting stream rendering for stream ${streamId}`);
    
    // Hide placeholder, show video
    if (placeholder) placeholder.style.display = 'none';
    imgElement.style.display = 'block';
    
    // Clear any existing interval
    if (streamIntervals[streamId]) {
        clearInterval(streamIntervals[streamId]);
        streamIntervals[streamId] = null;
    }
    
    // Use frame polling approach which is more reliable across browsers
    // MJPEG streams can be problematic with img.onerror in some browsers
    streamStates[streamId] = 'polling';
    startFramePolling(streamId);
}

function startFramePolling(streamId) {
    const imgElement = document.getElementById(`stream${streamId}`);
    const placeholder = document.getElementById(`placeholder${streamId}`);
    
    if (!imgElement) return;
    
    console.log(`Stream ${streamId}: Starting frame polling`);
    
    // Clear any existing interval
    if (streamIntervals[streamId]) {
        clearInterval(streamIntervals[streamId]);
    }
    
    let consecutiveErrors = 0;
    const maxConsecutiveErrors = 30; // About 3 seconds of errors
    
    function pollFrame() {
        if (streamStates[streamId] === false) {
            console.log(`Stream ${streamId}: Stopping frame polling (stream stopped)`);
            if (streamIntervals[streamId]) {
                clearInterval(streamIntervals[streamId]);
                streamIntervals[streamId] = null;
            }
            return;
        }
        
        const frameUrl = `${API_BASE}/stream/${streamId}/frame?t=${Date.now()}`;
        
        // Create temporary image to test load
        const tempImg = new Image();
        
        tempImg.onload = () => {
            consecutiveErrors = 0;
            streamRetryCount[streamId] = 0;
            
            // Update the actual image element
            imgElement.src = tempImg.src;
            imgElement.style.display = 'block';
            
            if (placeholder) {
                placeholder.style.display = 'none';
            }
        };
        
        tempImg.onerror = () => {
            consecutiveErrors++;
            console.warn(`Stream ${streamId}: Frame polling error (${consecutiveErrors}/${maxConsecutiveErrors})`);
            
            if (consecutiveErrors >= maxConsecutiveErrors) {
                console.error(`Stream ${streamId}: Too many errors, checking stream status...`);
                checkStreamStatus(streamId);
            }
        };
        
        tempImg.src = frameUrl;
    }
    
    // Start polling at ~15 FPS for smooth playback
    streamIntervals[streamId] = setInterval(pollFrame, 66);
    
    // Initial poll
    pollFrame();
}

async function checkStreamStatus(streamId) {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        
        const stream = data.streams?.find(s => s.stream_id === streamId);
        
        if (!stream || !stream.processing) {
            console.log(`Stream ${streamId}: Stream is not active, stopping polling`);
            stopStreamRendering(streamId);
        } else {
            // Stream is still active, continue polling
            streamRetryCount[streamId]++;
            
            if (streamRetryCount[streamId] >= MAX_RETRY_COUNT) {
                console.error(`Stream ${streamId}: Max retries exceeded`);
                showNotification('error', `Stream ${streamId + 1} connection lost. Please restart the stream.`);
                stopStreamRendering(streamId);
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
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, email, password, full_name })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || 'Signup failed');
                }

                // Store token and user data
                authToken = data.access_token;
                currentUser = data.user;
                
                // Save to localStorage
                localStorage.setItem('authToken', authToken);
                localStorage.setItem('currentUser', JSON.stringify(currentUser));

                // Update UI
                updateAuthUI();
                hideAuthModal();
                showToast(`Welcome to TMS, ${currentUser.username}!`, 'success');
                
                // Redirect to analytics page
                navigateTo('analytics');
                
            } catch (error) {
                showAuthError(error.message);
            } finally {
                setButtonLoading('signupSubmitBtn', false);
            }
        }

        async function logout() {
            try {
                if (authToken) {
                    await fetch(`${API_BASE_URL}/api/auth/logout`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${authToken}`
                        }
                    });
                }
            } catch (error) {
                console.error('Logout error:', error);
            }

            // Clear local storage and state
            localStorage.removeItem('authToken');
            localStorage.removeItem('currentUser');
            authToken = null;
            currentUser = null;

            
            // Stop all streams
            for (let i = 0; i < 4; i++) {
                if (streamPollIntervals[i]) {
                    clearInterval(streamPollIntervals[i]);
                    delete streamPollIntervals[i];
                }
            }
            
            // Update UI
            updateAuthUI();
            closeUserDropdown();
            showToast('You have been logged out', 'success');
            
            // Redirect to public landing page
            window.location.href = '/';
        }

        function updateAuthUI() {
            const loginBtn = document.getElementById('loginBtn');
            const userDropdown = document.getElementById('userDropdown');
            const authRequiredLinks = document.querySelectorAll('.auth-required');
            const mainContent = document.querySelector('.main-content');
            
            if (currentUser && authToken) {
                // User is logged in
                if (loginBtn) loginBtn.classList.add('hidden');
                if (userDropdown) userDropdown.classList.remove('hidden');
                
                authRequiredLinks.forEach(link => {
                    link.style.display = 'flex';
                });
                
                // Update user info
                if (currentUser.username) {
                    const initials = currentUser.username.substring(0, 2).toUpperCase();
                    const avatarEl = document.getElementById('userAvatar');
                    if (avatarEl) avatarEl.textContent = initials;
                }
                const nameEl = document.getElementById('userDisplayName');
                const emailEl = document.getElementById('userDisplayEmail');
                if (nameEl) nameEl.textContent = currentUser.full_name || currentUser.username || 'User';
                if (emailEl) emailEl.textContent = currentUser.email || '';

                // Unlock the dashboard content
                if (mainContent) mainContent.classList.remove('blurred');
            } else {
                // User is not logged in
                if (loginBtn) loginBtn.classList.remove('hidden');
                if (userDropdown) userDropdown.classList.add('hidden');
                
                authRequiredLinks.forEach(link => {
                    link.style.display = 'none';
                });

                // Lock/blur the dashboard content until login
                if (mainContent) mainContent.classList.add('blurred');
            }
        }

        function toggleUserDropdown() {
            const dropdown = document.getElementById('userDropdown');
            dropdown.classList.toggle('active');
        }

        function closeUserDropdown() {
            document.getElementById('userDropdown').classList.remove('active');
        }

        // Close dropdown when clicking outside
        document.addEventListener('click', function(event) {
            const dropdown = document.getElementById('userDropdown');
            if (dropdown && !dropdown.contains(event.target)) {
                closeUserDropdown();
            }
        });

        // Close auth modal when clicking outside
        document.getElementById('authModal').addEventListener('click', function(event) {
            if (event.target === this) {
                hideAuthModal();
            }
        });

        // Check for existing auth on page load
        function checkExistingAuth() {
            const savedToken = localStorage.getItem('authToken');
            const savedUser = localStorage.getItem('currentUser');
            
            if (savedToken && savedUser) {
                authToken = savedToken;
                currentUser = JSON.parse(savedUser);
                
                // Verify token is still valid
                verifyToken();
                
                // Ensure dashboard UI is in authenticated state
                updateAuthUI();
            } else {
                // Not authenticated: lock dashboard and prompt for login
                updateAuthUI();
                showToast('Please login to access the dashboard', 'error');
                showAuthModal();
            }
        }

        async function verifyToken() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/auth/verify`, {
                    headers: {
                        'Authorization': `Bearer ${authToken}`
                    }
                });
                
                if (!response.ok) {
                    // Token is invalid, clear auth
                    localStorage.removeItem('authToken');
                    localStorage.removeItem('currentUser');
                    authToken = null;
                    currentUser = null;
                    updateAuthUI();
                    return false;
                }
                return true;
            } catch (error) {
                console.error('Token verification error:', error);
                return false;
            }
        }

        // Navigate to a page
        function navigateTo(pageId) {
            // Check authentication for protected pages
            const authRequiredPages = ['analytics', 'monitoring', 'violations', 'about'];
            
            if (authRequiredPages.includes(pageId) && (!currentUser || !authToken)) {
                showToast('Please login to access this page', 'error');
                showAuthModal();
                return;
            }
            
            // Update navigation
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            
            const navLink = document.querySelector(`[data-page="${pageId}"]`);
            if (navLink) navLink.classList.add('active');
            
            const targetPage = document.getElementById(pageId);
            if (targetPage) {
                targetPage.classList.add('active');
            }
            
            closeUserDropdown();
            
            // Load page-specific data
            if (pageId === 'violations') {
                updateViolationsPage();
            }
            
            if (pageId === 'analytics' || pageId === 'monitoring') {
                updateSystemStats();
            }
        }

        // ==================== CHARTS (ANALYTICS) ====================
        function initCharts() {
            const ctxViolationsByType = document.getElementById('violationsByTypeChart');
            const ctxStreamsVehicles = document.getElementById('streamsVehiclesChart');
            const ctxViolationsOverTime = document.getElementById('violationsOverTimeChart');

            if (ctxViolationsByType && window.Chart) {
                violationsByTypeChart = new Chart(ctxViolationsByType, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Violations',
                            data: [],
                            backgroundColor: 'rgba(6, 182, 212, 0.6)',
                            borderColor: 'rgba(6, 182, 212, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: '#cbd5e1' } },
                            y: { ticks: { color: '#cbd5e1' }, beginAtZero: true }
                        }
                    }
                });
            }

            if (ctxStreamsVehicles && window.Chart) {
                streamsVehiclesChart = new Chart(ctxStreamsVehicles, {
                    type: 'doughnut',
                    data: {
                        labels: ['Active Streams', 'Total Vehicles'],
                        datasets: [{
                            data: [0, 0],
                            backgroundColor: [
                                'rgba(16, 185, 129, 0.7)',
                                'rgba(59, 130, 246, 0.7)'
                            ],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                labels: { color: '#cbd5e1' }
                            }
                        }
                    }
                });
            }

            if (ctxViolationsOverTime && window.Chart) {
                violationsOverTimeChart = new Chart(ctxViolationsOverTime, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Violations',
                            data: [],
                            borderColor: 'rgba(239, 68, 68, 1)',
                            backgroundColor: 'rgba(239, 68, 68, 0.3)',
                            tension: 0.3,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { labels: { color: '#cbd5e1' } }
                        },
                        scales: {
                            x: { ticks: { color: '#cbd5e1' } },
                            y: { ticks: { color: '#cbd5e1' }, beginAtZero: true }
                        }
                    }
                });
            }
        }

        function updateChartsFromStats(data) {
            if (!data) return;

            // Violations by type (from summary)
            if (violationsByTypeChart && data.violation_summary) {
                const labels = Object.keys(data.violation_summary);
                const values = labels.map(k => data.violation_summary[k] || 0);
                violationsByTypeChart.data.labels = labels.map(l => l.replace('_', ' ').toUpperCase());
                violationsByTypeChart.data.datasets[0].data = values;
                violationsByTypeChart.update();
            }

            // Streams & vehicles
            if (streamsVehiclesChart) {
                streamsVehiclesChart.data.datasets[0].data = [
                    data.active_streams || 0,
                    data.total_vehicles || 0
                ];
                streamsVehiclesChart.update();
            }

            // Violations over time (simple bucketing by time string)
            if (violationsOverTimeChart && Array.isArray(data.violations)) {
                const buckets = {};
                data.violations.forEach(v => {
                    if (!v.timestamp) return;
                    const d = new Date(v.timestamp);
                    if (isNaN(d.getTime())) return;
                    const key = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    buckets[key] = (buckets[key] || 0) + 1;
                });
                const labels = Object.keys(buckets).sort();
                const values = labels.map(k => buckets[k]);
                violationsOverTimeChart.data.labels = labels;
                violationsOverTimeChart.data.datasets[0].data = values;
                violationsOverTimeChart.update();
            }
        }

        // ==================== UTILITY FUNCTIONS ====================
        
        function getAuthHeaders() {
            const headers = {
                'Content-Type': 'application/json'
            };
            if (authToken) {
                headers['Authorization'] = `Bearer ${authToken}`;
            }
            return headers;
        }
        
        function showToast(message, type = 'success ') {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }

        function formatTime(isoString) {
            if (!isoString) return 'N/A';
            const date = new Date(isoString);
            return date.toLocaleTimeString();
        }

        function formatSpeed(speed) {
            return speed ? `${speed.toFixed(1)} km/h` : 'N/A';
        }

        // ==================== STREAM MANAGEMENT ====================
        function startStream(index) {
            const url = document.getElementById('streamUrl' + index).value;
            if (!url) {
                showToast('Please enter a stream URL', 'error');
                return;
            }

            console.log(`Starting stream ${index} with URL:`, url);
            showToast(`Starting stream ${index + 1}...`, 'info');

            fetch(`${API_BASE_URL}/api/start-stream/${index}?stream_url=${encodeURIComponent(url)}`, {
                method: 'POST',
                headers: getAuthHeaders()
            })
            .then(async r => {
                const data = await r.json();
                if (!r.ok) {
                    throw new Error(data.detail || `Server returned ${r.status}`);
                }
                return data;
            })
            .then(data => {
                console.log(`Stream ${index} started successfully:`, data);
                showToast(`Stream ${index + 1} started successfully!`, 'success');
                updateStreamStatus(index);
                pollStreamFrame(index);
            })
            .catch(err => {
                console.error(`Failed to start stream ${index}:`, err);
                showToast(`Failed to start stream: ${err.message}`, 'error');
            });
        }

        function stopStream(index) {
            fetch(`${API_BASE_URL}/api/stop-stream/${index}`, { 
                method: 'POST',
                headers: getAuthHeaders()
            })
            .then(r => r.json())
            .then(data => {
                showToast(`Stream ${index + 1} stopped`, 'success');
                document.getElementById('stream' + index).style.display = 'none';
                document.getElementById('placeholder' + index).style.display = 'flex';
                document.getElementById('status' + index).textContent = 'Inactive';
                document.getElementById('status' + index).className = 'status-badge inactive';
                
                if (streamPollIntervals[index]) {
                    clearInterval(streamPollIntervals[index]);
                    delete streamPollIntervals[index];
                }
            })
            .catch(err => showToast(`Failed to stop stream: ${err.message}`, 'error'));
        }

        function updateStreamStatus(index) {
            // This will be called after stream starts
            document.getElementById('status' + index).textContent = 'Active';
            document.getElementById('status' + index).className = 'status-badge active';
        }

        function pollStreamFrame(index) {
            if (streamPollIntervals[index]) {
                clearInterval(streamPollIntervals[index]);
            }

            streamPollIntervals[index] = setInterval(() => {
                fetch(`${API_BASE_URL}/stream/${index}/frame`)
                    .then(r => r.blob())
                    .then(blob => {
                        const url = URL.createObjectURL(blob);
                        const img = document.getElementById('stream' + index);
                        const placeholder = document.getElementById('placeholder' + index);
                        img.src = url;
                        img.style.display = 'block';
                        placeholder.style.display = 'none';
                    })
                    .catch(err => console.error(`Error polling stream ${index}:`, err));
            }, STREAM_POLL_INTERVAL);
        }

        // ==================== FILE UPLOAD ====================
        function setupFileUpload(index) {
            const fileInput = document.getElementById('videoFile' + index);
            const uploadBox = document.querySelector(`[for="videoFile${index}"]`).parentElement.querySelector('.stream-upload-box');
            const uploadLabel = document.querySelector(`[for="videoFile${index}"]`);
            const selectedFile = document.getElementById('selectedFile' + index);
            const fileName = document.getElementById('fileName' + index);

            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    fileName.textContent = file.name;
                    selectedFile.style.display = 'flex';
                }
            });

            uploadLabel.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadBox.style.borderColor = 'var(--accent)';
                uploadBox.style.background = 'rgba(6, 182, 212, 0.15)';
            });

            uploadLabel.addEventListener('dragleave', () => {
                uploadBox.style.borderColor = 'var(--border-color)';
                uploadBox.style.background = 'rgba(6, 182, 212, 0.02)';
            });

            uploadLabel.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadBox.style.borderColor = 'var(--border-color)';
                uploadBox.style.background = 'rgba(6, 182, 212, 0.02)';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    const event = new Event('change', { bubbles: true });
                    fileInput.dispatchEvent(event);
                }
            });
        }

        function clearStreamFile(index) {
            document.getElementById('videoFile' + index).value = '';
            document.getElementById('selectedFile' + index).style.display = 'none';
        }

        function uploadStreamVideo(index) {
            const fileInput = document.getElementById('videoFile' + index);
            const file = fileInput.files[0];
            
            if (!file) {
                showToast('Please select a video file', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            const headers = {};
            if (authToken) {
                headers['Authorization'] = `Bearer ${authToken}`;
            }

            fetch(`${API_BASE_URL}/api/upload-video/${index}`, {
                method: 'POST',
                headers: headers,
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                showToast(`Video uploading and processing started for Stream ${index + 1}`, 'success');
                clearStreamFile(index);
                updateStreamStatus(index);
                pollStreamFrame(index);
            })
            .catch(err => showToast(`Upload failed: ${err.message}`, 'error'));
        }

        // ==================== STATS AND VIOLATIONS ====================
        async function updateSystemStats() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/stats`);
                const data = await response.json();
                systemStats = data;

                // Update analytics summary cards (all-time baseline)
                document.getElementById('activeStreams').textContent = data.active_streams;
                document.getElementById('totalVehicles').textContent = data.total_vehicles;
                document.getElementById('totalViolations').textContent = data.total_violations;
                document.getElementById('notificationBadge').textContent = data.total_violations;
                
                // Count violation types
                const summary = data.violation_summary || {};
                document.getElementById('speedViolations').textContent = summary.speed || 0;

                // Update recent violations table (all-time baseline)
                updateViolationsTable((data.violations || []).slice(0, 10));

                // Update analytics charts with baseline stats
                updateChartsFromStats(data);

            } catch (err) {
                console.error('Error fetching stats:', err);
            }
        }

        function updateViolationsTable(violations) {
            const tbody = document.getElementById('violationsTableBody');
            
            if (!violations || violations.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="empty-state"><i class="fas fa-inbox"></i><p>No violations</p></td></tr>';
                return;
            }

            tbody.innerHTML = violations.map(v => `
                <tr>
                    <td>Stream ${v.stream_id}</td>
                    <td>${v.violation_type.replace('_', ' ').toUpperCase()}</td>
                    <td>${formatSpeed(v.speed_kmh)}</td>
                    <td>${formatTime(v.timestamp)}</td>
                </tr>
            `).join('');
        }

        async function updateViolationsPage() {
            const dateFilterSelect = document.getElementById('violationDateFilter');
            const dateRange = dateFilterSelect ? dateFilterSelect.value : 'all';
            const queryRange = dateRange === 'all' ? '' : `&date_range=${encodeURIComponent(dateRange)}`;

            // Try to fetch from database first with date filter
            try {
                const response = await fetch(`${API_BASE_URL}/api/db/violations?limit=200${queryRange}`, {
                    headers: getAuthHeaders()
                });
                const data = await response.json();
                
                if (data.violations && data.violations.length > 0) {
                    // Count violations by type
                    const counts = {
                        speed: 0,
                        red_light: 0,
                        stop_line: 0,
                        lane_change: 0,
                        unsafe_distance: 0
                    };
                    
                    data.violations.forEach(v => {
                        if (counts.hasOwnProperty(v.violation_type)) {
                            counts[v.violation_type]++;
                        }
                    });

                    document.getElementById('speedViolationCount').textContent = counts.speed;
                    document.getElementById('redLightCount').textContent = counts.red_light;
                    document.getElementById('stopLineCount').textContent = counts.stop_line || counts.unsafe_distance;
                    document.getElementById('laneChangeCount').textContent = counts.lane_change;

                    const tbody = document.getElementById('allViolationsTable');
                    tbody.innerHTML = data.violations.map(v => `
                        <tr>
                            <td>Stream ${v.stream_id}</td>
                            <td>${v.violation_type.replace('_', ' ').toUpperCase()}</td>
                            <td>${formatSpeed(v.speed_kmh)}</td>
                            <td>${v.signal_state || 'N/A'}</td>
                            <td>${formatTime(v.timestamp)}</td>
                        </tr>
                    `).join('');
                    return;
                }
            } catch (err) {
                console.error('Error fetching from DB:', err);
            }

            // Fallback to in-memory stats (also apply date filter on client side)
            if (!systemStats.violations) return;

            let filteredViolations = systemStats.violations;
            if (dateRange !== 'all') {
                const now = new Date();
                let startTime = null;
                let endTime = null;

                if (dateRange === 'yesterday') {
                    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                    startTime = new Date(todayStart.getTime() - 24 * 60 * 60 * 1000);
                    endTime = todayStart;
                } else if (dateRange === 'last_week') {
                    startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                } else if (dateRange === 'last_month') {
                    startTime = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                } else if (dateRange === 'last_year') {
                    startTime = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
                }

                if (startTime) {
                    filteredViolations = filteredViolations.filter(v => {
                        if (!v.timestamp) return false;
                        const d = new Date(v.timestamp);
                        if (isNaN(d.getTime())) return false;
                        if (endTime) {
                            return d >= startTime && d < endTime;
                        }
                        return d >= startTime;
                    });
                }
            }

            const summary = systemStats.violation_summary || {};
            document.getElementById('speedViolationCount').textContent = summary.speed || 0;
            document.getElementById('redLightCount').textContent = summary.red_light || 0;
            document.getElementById('stopLineCount').textContent = summary.stop_line || 0;
            document.getElementById('laneChangeCount').textContent = summary.lane_change || 0;

            const tbody = document.getElementById('allViolationsTable');
            if (!filteredViolations || filteredViolations.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state"><i class="fas fa-inbox"></i><p>No violations</p></td></tr>';
                return;
            }

            tbody.innerHTML = filteredViolations.map(v => `
                <tr>
                    <td>Stream ${v.stream_id}</td>
                    <td>${v.violation_type.replace('_', ' ').toUpperCase()}</td>
                    <td>${formatSpeed(v.speed_kmh)}</td>
                    <td>${v.signal_state || 'N/A'}</td>
                    <td>${formatTime(v.timestamp)}</td>
                </tr>
            `).join('');
        }

        // ==================== WEBSOCKET CONNECTION ====================
        function connectWebSocket() {
            // Use the configured WebSocket URL
            const wsUrl = `${WS_BASE_URL}/ws`;
            console.log('🔌 Connecting to WebSocket:', wsUrl);
            
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                showToast('Connected to backend');
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'violation') {
                    // New violation detected
                    updateSystemStats();
                    showToast(`Violation detected: ${message.data.violation_type}`, 'error');
                } else if (message.type === 'stats_update') {
                    systemStats = message.data;
                    updateSystemStats();
                }
            };

            ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                showToast('Connection error', 'error');
            };

            ws.onclose = () => {
                console.log('⚠️ WebSocket disconnected, reconnecting in 3s...');
                setTimeout(connectWebSocket, 3000);
            };
        }

        // ==================== ANALYTICS RANGE ====================
        async function updateAnalyticsRange() {
            const select = document.getElementById('analyticsDateFilter');
            analyticsDateRange = select ? select.value : 'all';

            const queryRange = analyticsDateRange === 'all'
                ? ''
                : `&date_range=${encodeURIComponent(analyticsDateRange)}`;

            // Try DB first for precise history
            try {
                const response = await fetch(`${API_BASE_URL}/api/db/violations?limit=500${queryRange}`, {
                    headers: getAuthHeaders()
                });
                const data = await response.json();

                if (data.violations && data.violations.length > 0) {
                    const violations = data.violations;

                    // Build summary from filtered history
                    const summary = {
                        speed: 0,
                        red_light: 0,
                        stop_line: 0,
                        lane_change: 0,
                        unsafe_distance: 0
                    };
                    violations.forEach(v => {
                        if (summary.hasOwnProperty(v.violation_type)) {
                            summary[v.violation_type]++;
                        }
                    });

                    // Update summary cards (filtered history for violations)
                    document.getElementById('totalViolations').textContent = violations.length;
                    document.getElementById('speedViolations').textContent = summary.speed || 0;

                    // Update recent violations table from filtered data
                    updateViolationsTable(violations.slice(0, 10));

                    // Update charts using filtered data
                    const statsLike = Object.assign({}, systemStats, {
                        violation_summary: summary,
                        violations: violations
                    });
                    updateChartsFromStats(statsLike);
                    return;
                }
            } catch (err) {
                console.error('Error updating analytics range from DB:', err);
            }

            // Fallback: filter in-memory violations
            if (!systemStats.violations) return;

            let filteredViolations = systemStats.violations;
            if (analyticsDateRange !== 'all') {
                const now = new Date();
                let startTime = null;
                let endTime = null;

                if (analyticsDateRange === 'yesterday') {
                    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                    startTime = new Date(todayStart.getTime() - 24 * 60 * 60 * 1000);
                    endTime = todayStart;
                } else if (analyticsDateRange === 'last_week') {
                    startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                } else if (analyticsDateRange === 'last_month') {
                    startTime = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                } else if (analyticsDateRange === 'last_year') {
                    startTime = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
                }

                if (startTime) {
                    filteredViolations = filteredViolations.filter(v => {
                        if (!v.timestamp) return false;
                        const d = new Date(v.timestamp);
                        if (isNaN(d.getTime())) return false;
                        if (endTime) {
                            return d >= startTime && d < endTime;
                        }
                        return d >= startTime;
                    });
                }
            }

            const summary = {
                speed: 0,
                red_light: 0,
                stop_line: 0,
                lane_change: 0,
                unsafe_distance: 0
            };
            filteredViolations.forEach(v => {
                if (summary.hasOwnProperty(v.violation_type)) {
                    summary[v.violation_type]++;
                }
            });

            document.getElementById('totalViolations').textContent = filteredViolations.length;
            document.getElementById('speedViolations').textContent = summary.speed || 0;

            updateViolationsTable(filteredViolations.slice(0, 10));

            const statsLike = Object.assign({}, systemStats, {
                violation_summary: summary,
                violations: filteredViolations
            });
            updateChartsFromStats(statsLike);
        }

        // ==================== INITIALIZATION ====================
        document.addEventListener('DOMContentLoaded', function() {
            console.log('✅ DOM Content Loaded - Initializing...');
            
            // Setup navigation links
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const pageId = this.getAttribute('data-page');
                    console.log('Navigation clicked:', pageId);
                    navigateTo(pageId);
                });
            });

            // Initialize analytics charts
            initCharts();
            
            // Check for existing authentication
            checkExistingAuth();
            
            // Setup file uploads for all streams
            for (let i = 0; i < 4; i++) {
                setupFileUpload(i);
            }

            // Initial stats load
            updateSystemStats();

            // Start periodic updates
            pollInterval = setInterval(updateSystemStats, POLL_INTERVAL);

            // Connect WebSocket for real-time updates
            connectWebSocket();

            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {
                if (pollInterval) clearInterval(pollInterval);
                Object.values(streamPollIntervals).forEach(interval => clearInterval(interval));
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                // Escape to close modal
                if (e.key === 'Escape') {
                    hideAuthModal();
                    closeUserDropdown();
                }
            });
        });