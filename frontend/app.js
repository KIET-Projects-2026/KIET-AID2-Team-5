// API Configuration
const API_BASE = 'http://localhost:8000';
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
    } catch (error) {
        console.error(`Stream ${streamId}: Failed to check status:`, error);
    }
}

function stopStreamRendering(streamId) {
    const imgElement = document.getElementById(`stream${streamId}`);
    const placeholder = document.getElementById(`placeholder${streamId}`);
    
    console.log(`Stopping stream rendering for stream ${streamId}`);
    
    // Stop polling interval
    if (streamIntervals[streamId]) {
        clearInterval(streamIntervals[streamId]);
        streamIntervals[streamId] = null;
    }
    
    streamStates[streamId] = false;
    streamRetryCount[streamId] = 0;
    
    if (imgElement) {
        imgElement.src = '';
        imgElement.style.display = 'none';
    }
    
    if (placeholder) {
        placeholder.style.display = 'flex';
    }
}

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    connectWebSocket();
    startDataRefresh();
});

function initializeApp() {
    console.log('Initializing PTU Traffic Monitoring System...');
    switchPage('home');
}

// ==================== NAVIGATION ====================

function setupEventListeners() {
    // Navbar links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.getAttribute('data-page');
            switchPage(page);
            
            // Update active states
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
    
    // Sidebar menu items
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.getAttribute('data-page');
            switchPage(page);
            
            // Update active states
            document.querySelectorAll('.menu-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // Update navbar active state
            document.querySelectorAll('.nav-link').forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('data-page') === page) {
                    link.classList.add('active');
                }
            });
        });
    });
}

function switchPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(`${pageName}Page`);
    if (targetPage) {
        targetPage.classList.add('active');
        
        // Load page-specific data
        if (pageName === 'violations') {
            loadViolations();
        }
    }
}

// ==================== WEBSOCKET CONNECTION ====================

function connectWebSocket() {
    const wsUrl = 'ws://localhost:8000/ws';
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            updateConnectionStatus(true);
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        ws.onclose = () => {
            console.log('WebSocket disconnected');
            updateConnectionStatus(false);
            // Reconnect after 3 seconds
            setTimeout(connectWebSocket, 3000);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus(false);
        };
    } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        updateConnectionStatus(false);
    }
}

function handleWebSocketMessage(data) {
    if (data.type === 'stats_update') {
        updateDashboard(data.data);
    } else if (data.type === 'violation') {
        addNewViolation(data.data);
    }
}

function addNewViolation(violation) {
    console.log('New violation detected:', violation);
    
    // Add to violations array
    allViolations.unshift(violation);
    
    // Map violation types to display info
    const violationTypeMap = {
        'speed': { label: 'Speed', isSpeed: true },
        'red_light': { label: 'Red Light', isSpeed: false },
        'stop_line': { label: 'Stop Line', isSpeed: false },
        'lane_change': { label: 'Lane Change', isSpeed: false },
        'wrong_lane': { label: 'Wrong Lane', isSpeed: false },
        'unsafe_distance': { label: 'Unsafe Distance', isSpeed: false }
    };
    
    const vType = violation.violation_type || 'speed';
    const typeInfo = violationTypeMap[vType] || { label: 'Violation', isSpeed: false };
    
    let details = `${violation.vehicle_class || 'Vehicle'}`;
    if (vType === 'speed' && violation.speed_kmh) {
        details += ` - ${violation.speed_kmh} km/h`;
        if (violation.speed_limit) {
            details += ` (Limit: ${violation.speed_limit} km/h)`;
        }
    } else if ((vType === 'red_light' || vType === 'stop_line') && violation.signal_state) {
        details += ` - Crossed during ${violation.signal_state} light`;
    }
    
    showNotification('error', `${typeInfo.label} Violation Detected! Stream ${(violation.stream_id || 0) + 1} - ${details}`);
    
    // Update notification badge
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        const current = parseInt(badge.textContent) || 0;
        badge.textContent = current + 1;
    }
    
    // Update violation counters
    const totalElement = document.getElementById('totalViolations');
    if (totalElement) {
        const current = parseInt(totalElement.textContent) || 0;
        updateElement('totalViolations', current + 1);
    }
    
    if (typeInfo.isSpeed) {
        const speedElement = document.getElementById('speedViolations');
        if (speedElement) {
            const current = parseInt(speedElement.textContent) || 0;
            updateElement('speedViolations', current + 1);
        }
    } else {
        const signalElement = document.getElementById('signalViolations');
        if (signalElement) {
            const current = parseInt(signalElement.textContent) || 0;
            updateElement('signalViolations', current + 1);
        }
    }
    
    // Update violations page summary if on that page
    const violationTotal = document.getElementById('violationTotal');
    if (violationTotal) {
        const current = parseInt(violationTotal.textContent) || 0;
        updateElement('violationTotal', current + 1);
    }
    
    if (typeInfo.isSpeed) {
        const speedCount = document.getElementById('violationSpeed');
        if (speedCount) {
            const current = parseInt(speedCount.textContent) || 0;
            updateElement('violationSpeed', current + 1);
        }
    } else {
        const signalCount = document.getElementById('violationSignal');
        if (signalCount) {
            const current = parseInt(signalCount.textContent) || 0;
            updateElement('violationSignal', current + 1);
        }
    }
    
    // Check if violation is today
    const today = new Date().toDateString();
    const vDate = new Date(violation.timestamp).toDateString();
    if (vDate === today) {
        const todayCount = document.getElementById('violationToday');
        if (todayCount) {
            const current = parseInt(todayCount.textContent) || 0;
            updateElement('violationToday', current + 1);
        }
    }
    
    // Refresh violations if on violations page
    if (document.getElementById('violationsPage').classList.contains('active')) {
        updateViolationsTable(allViolations);
    }
    
    // Refresh recent violations on home page
    updateRecentViolations(allViolations.slice(0, 5));
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('statusIndicator');
    const text = document.getElementById('statusText');
    
    if (connected) {
        indicator.style.background = 'var(--primary-green)';
        text.textContent = 'Connected';
    } else {
        indicator.style.background = 'var(--primary-red)';
        text.textContent = 'Disconnected';
    }
}

// ==================== DATA REFRESH ====================

async function startDataRefresh() {
    // Initial load
    await fetchStats();
    
    // Refresh every 2 seconds
    setInterval(fetchStats, 2000);
}

async function fetchStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

// ==================== DASHBOARD UPDATES ====================

function updateDashboard(data) {
    // Update surveillance status cards
    const totalStreams = data.total_streams || 4;
    const activeStreams = data.active_streams || 0;
    const inactiveStreams = totalStreams - activeStreams;
    const maintenanceStreams = 0; // Can be calculated based on stream status
    
    updateElement('totalStreams', totalStreams);
    updateElement('activeStreams', activeStreams);
    updateElement('inactiveStreams', inactiveStreams);
    updateElement('maintenanceStreams', maintenanceStreams);
    
    // Update traffic analytics
    const violationSummary = data.violation_summary || {};
    const speedViolations = (violationSummary.speed || 0);
    const redLightViolations = (violationSummary.red_light || 0);
    const stopLineViolations = (violationSummary.stop_line || 0);
    const laneViolations = (violationSummary.lane_change || 0);
    const wrongLaneViolations = (violationSummary.wrong_lane || 0);
    const distanceViolations = (violationSummary.unsafe_distance || 0);
    
    const totalViolations = data.total_violations || 0;
    const signalViolations = redLightViolations + stopLineViolations;
    
    updateElement('totalVehicles', data.total_vehicles || 0);
    updateElement('totalViolations', totalViolations);
    updateElement('speedViolations', speedViolations);
    updateElement('signalViolations', signalViolations);
    
    // Update notification badge
    updateElement('notificationBadge', totalViolations);
    
    // Update stream-specific data
    if (data.streams) {
        data.streams.forEach(stream => {
            updateStreamData(stream);
        });
    }
    
    // Update recent violations
    if (data.violations) {
        updateRecentViolations(data.violations.slice(0, 5));
        allViolations = data.violations;
    }
    
    // Update violations page if active
    if (document.getElementById('violationsPage').classList.contains('active')) {
        updateViolationsTable(data.violations || []);
    }
}

function updateStreamData(stream) {
    const streamId = stream.stream_id;
    
    // Update stream status badge
    const statusElement = document.getElementById(`status${streamId}`);
    if (statusElement) {
        if (stream.processing) {
            statusElement.textContent = 'Active';
            statusElement.classList.add('active');
        } else {
            statusElement.textContent = 'Inactive';
            statusElement.classList.remove('active');
        }
    }
    
    // Update video feed
    const videoFeed = document.getElementById(`stream${streamId}`);
    const placeholder = document.getElementById(`placeholder${streamId}`);
    const liveBadge = document.getElementById(`live${streamId}`);
    
    if (stream.processing && videoFeed) {
        // Start stream rendering if not already started
        // Check for false specifically (could be 'polling' or other truthy value)
        if (streamStates[streamId] === false || streamStates[streamId] === undefined) {
            console.log(`Dashboard: Starting stream ${streamId} rendering`);
            // Small delay to ensure backend is ready
            setTimeout(() => {
                startStreamRendering(streamId);
            }, 500);
        }
        if (liveBadge) liveBadge.style.display = 'flex';
    } else if (videoFeed) {
        // Stop stream rendering if stream is not processing
        if (streamStates[streamId] && streamStates[streamId] !== false) {
            console.log(`Dashboard: Stopping stream ${streamId} rendering`);
            stopStreamRendering(streamId);
        }
        if (liveBadge) liveBadge.style.display = 'none';
    }
    
    // Update stream stats
    const statsContainer = document.getElementById(`stats${streamId}`);
    if (statsContainer) {
        const statElements = statsContainer.querySelectorAll('.stat span');
        if (statElements.length >= 3) {
            statElements[0].textContent = stream.total_vehicles || 0;
            statElements[1].textContent = Math.round(stream.average_speed || 0);
            
            // Sum all violation types
            const violationCounts = stream.violation_counts || {};
            const totalStreamViolations = Object.values(violationCounts).reduce((sum, count) => sum + count, 0);
            statElements[2].textContent = totalStreamViolations;
        }
    }
}

function updateRecentViolations(violations) {
    const container = document.getElementById('recentViolationsList');
    
    if (!violations || violations.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-check-circle"></i>
                <p>No violations detected</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = violations.map(v => {
        // Map violation types to display info
        const violationTypeMap = {
            'speed': { icon: 'fa-tachometer-alt', label: 'Speed' },
            'red_light': { icon: 'fa-traffic-light', label: 'Red Light' },
            'stop_line': { icon: 'fa-hand-paper', label: 'Stop Line' },
            'lane_change': { icon: 'fa-road', label: 'Lane Change' },
            'wrong_lane': { icon: 'fa-directions', label: 'Wrong Lane' },
            'unsafe_distance': { icon: 'fa-compress-arrows-alt', label: 'Unsafe Distance' }
        };
        
        const vType = v.violation_type || 'speed';
        const typeInfo = violationTypeMap[vType] || { icon: 'fa-exclamation-triangle', label: 'Violation' };
        const icon = typeInfo.icon;
        const label = typeInfo.label;
        
        const title = `${label} Violation`;
        let details = `Stream ${(v.stream_id || 0) + 1} • ${v.vehicle_class || 'Vehicle'} • Track ID: ${v.track_id}`;
        
        if (vType === 'speed' && v.speed_kmh) {
            details += ` • ${v.speed_kmh} km/h`;
        }
        
        const time = formatTime(v.timestamp);
        const cssClass = vType === 'speed' ? 'speed' : 'signal';
        
        return `
            <div class="violation-item ${cssClass}">
                <div class="violation-info">
                    <div class="violation-icon">
                        <i class="fas ${icon}"></i>
                    </div>
                    <div class="violation-details">
                        <h4>${title}</h4>
                        <p>${details}</p>
                    </div>
                </div>
                <div class="violation-meta">
                    <span class="violation-time">${time}</span>
                </div>
            </div>
        `;
    }).join('');
}

// ==================== STREAM CONTROL ====================

async function startStream(streamId) {
    const urlInput = document.getElementById(`url${streamId}`);
    const streamUrl = urlInput.value.trim();
    
    if (!streamUrl) {
        showNotification('warning', 'Please enter a video URL or YouTube link');
        return;
    }
    
    try {
        showNotification('info', `Starting stream ${streamId + 1}...`);
        
        const response = await fetch(`${API_BASE}/api/start-stream/${streamId}?stream_url=${encodeURIComponent(streamUrl)}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log(`Stream ${streamId} started`);
            showNotification('success', `Stream ${streamId + 1} started successfully`);
            
            // Reset stream state to allow fresh connection
            streamStates[streamId] = false;
            streamRetryCount[streamId] = 0;
            
            // Give backend time to initialize, then start rendering
            setTimeout(() => {
                startStreamRendering(streamId);
            }, 1000);
        } else {
            const error = await response.json();
            showNotification('error', `Failed to start stream: ${error.detail}`);
        }
    } catch (error) {
        console.error('Error starting stream:', error);
        showNotification('error', 'Failed to start stream');
    }
}

async function stopStream(streamId) {
    try {
        const response = await fetch(`${API_BASE}/api/stop-stream/${streamId}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log(`Stream ${streamId} stopped`);
            showNotification('info', `Stream ${streamId + 1} stopped`);
            
            // Stop stream rendering
            stopStreamRendering(streamId);
            
            // Clear file input and related UI
            const fileInput = document.getElementById(`file${streamId}`);
            const fileName = document.getElementById(`fileName${streamId}`);
            const uploadBtn = document.getElementById(`uploadBtn${streamId}`);
            const startBtn = document.querySelector(`[onclick="startStream(${streamId})"]`);
            
            if (fileInput) fileInput.value = '';
            if (fileName) fileName.textContent = '';
            if (uploadBtn) uploadBtn.style.display = 'none';
            if (startBtn) startBtn.style.display = 'flex';
        }
    } catch (error) {
        console.error('Error stopping stream:', error);
        showNotification('error', 'Failed to stop stream');
    }
}

// ==================== VIDEO UPLOAD ====================

function handleFileSelect(streamId) {
    const fileInput = document.getElementById(`file${streamId}`);
    const fileName = document.getElementById(`fileName${streamId}`);
    const uploadBtn = document.getElementById(`uploadBtn${streamId}`);
    const startBtn = document.querySelector(`[onclick="startStream(${streamId})"]`);
    
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        fileName.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        
        // Show upload button and hide start button
        if (uploadBtn) uploadBtn.style.display = 'flex';
        if (startBtn) startBtn.style.display = 'none';
        
        // Clear URL input
        const urlInput = document.getElementById(`url${streamId}`);
        if (urlInput) urlInput.value = '';
    } else {
        fileName.textContent = '';
        if (uploadBtn) uploadBtn.style.display = 'none';
        if (startBtn) startBtn.style.display = 'flex';
    }
}

async function uploadVideo(streamId) {
    const fileInput = document.getElementById(`file${streamId}`);
    const uploadBtn = document.getElementById(`uploadBtn${streamId}`);
    
    if (!fileInput.files.length) {
        showNotification('error', 'Please select a video file first');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Show uploading status
        if (uploadBtn) {
            uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
            uploadBtn.disabled = true;
        }
        
        showNotification('info', `Uploading ${file.name}...`);
        
        const response = await fetch(`${API_BASE}/api/upload-video/${streamId}`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log(`Video uploaded and stream ${streamId} started:`, data);
            showNotification('success', `Video uploaded successfully! Stream ${streamId + 1} started.`);
            
            // Reset upload button
            if (uploadBtn) {
                uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Upload & Start';
                uploadBtn.disabled = false;
                uploadBtn.style.display = 'none';
            }
            
            // Show start button again
            const startBtn = document.querySelector(`[onclick="startStream(${streamId})"]`);
            if (startBtn) startBtn.style.display = 'flex';
            
            // Clear file input
            const fileName = document.getElementById(`fileName${streamId}`);
            if (fileName) fileName.textContent = '';
            if (fileInput) fileInput.value = '';
            
            // Reset stream state and start rendering
            streamStates[streamId] = false;
            streamRetryCount[streamId] = 0;
            
            // Give backend time to initialize processing, then start rendering
            setTimeout(() => {
                console.log(`Starting stream rendering after upload for stream ${streamId}`);
                startStreamRendering(streamId);
            }, 1500);
            
        } else {
            const error = await response.json();
            showNotification('error', `Upload failed: ${error.detail}`);
            
            // Reset upload button
            if (uploadBtn) {
                uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Upload & Start';
                uploadBtn.disabled = false;
            }
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        showNotification('error', 'Failed to upload video');
        
        // Reset upload button
        if (uploadBtn) {
            uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Upload & Start';
            uploadBtn.disabled = false;
        }
    }
}

// ==================== VIOLATIONS PAGE ====================

async function loadViolations() {
    try {
        const response = await fetch(`${API_BASE}/api/violations?limit=100`);
        const data = await response.json();
        
        // Update summary
        const violationSummary = data.violation_summary || {};
        const speedCount = violationSummary.speed || 0;
        const redLightCount = violationSummary.red_light || 0;
        const stopLineCount = violationSummary.stop_line || 0;
        const signalCount = redLightCount + stopLineCount;
        
        updateElement('violationTotal', data.total || 0);
        updateElement('violationSpeed', speedCount);
        updateElement('violationSignal', signalCount);
        
        // Calculate today's violations
        const today = new Date().toDateString();
        const todayViolations = (data.violations || []).filter(v => {
            const vDate = new Date(v.timestamp).toDateString();
            return vDate === today;
        });
        updateElement('violationToday', todayViolations.length);
        
        // Update table
        updateViolationsTable(data.violations || []);
    } catch (error) {
        console.error('Error loading violations:', error);
    }
}

function updateViolationsTable(violations) {
    const tbody = document.getElementById('violationsTableBody');
    
    if (!violations || violations.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="empty-state">
                    <i class="fas fa-check-circle"></i>
                    <p>No violations detected</p>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = violations.map(v => {
        // Map violation types to display labels
        const violationLabels = {
            'speed': 'Speed Limit',
            'red_light': 'Red Light',
            'stop_line': 'Stop Line',
            'lane_change': 'Lane Change',
            'wrong_lane': 'Wrong Lane',
            'unsafe_distance': 'Unsafe Distance'
        };
        
        const vType = v.violation_type || 'speed';
        const typeLabel = violationLabels[vType] || 'Violation';
        const cssClass = vType === 'speed' ? 'speed' : 'signal';
        
        let details = '';
        if (vType === 'speed' && v.speed_limit && v.speed_kmh) {
            details = `Over limit by ${(v.speed_kmh - v.speed_limit).toFixed(1)} km/h`;
        } else if (vType === 'red_light' || vType === 'stop_line') {
            details = `Crossed during ${v.signal_state || 'RED'} light`;
        } else if (vType === 'lane_change') {
            details = `Improper lane change to Lane ${v.lane || 'N/A'}`;
        } else if (vType === 'wrong_lane') {
            details = `Wrong lane driving detected`;
        } else if (vType === 'unsafe_distance') {
            details = `Following too closely`;
        } else {
            details = `Violation detected`;
        }
        
        return `
            <tr>
                <td>#${v.id}</td>
                <td>Stream ${(v.stream_id || 0) + 1}</td>
                <td><span class="violation-badge ${cssClass}">${typeLabel}</span></td>
                <td>${v.vehicle_class || 'Unknown'}</td>
                <td>${v.speed_kmh ? v.speed_kmh.toFixed(1) : 'N/A'} km/h</td>
                <td>${details}</td>
                <td>${formatTimestamp(v.timestamp)}</td>
                <td>
                    <button class="view-btn" onclick="viewViolation(${v.id})">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function applyFilters() {
    const typeFilter = document.getElementById('violationTypeFilter').value;
    const streamFilter = document.getElementById('violationStreamFilter').value;
    const vehicleFilter = document.getElementById('violationVehicleFilter').value;
    
    let filtered = [...allViolations];
    
    if (typeFilter !== 'all') {
        filtered = filtered.filter(v => v.violation_type === typeFilter);
    }
    
    if (streamFilter !== 'all') {
        filtered = filtered.filter(v => v.stream_id === parseInt(streamFilter));
    }
    
    if (vehicleFilter !== 'all') {
        filtered = filtered.filter(v => v.vehicle_class && v.vehicle_class.toLowerCase() === vehicleFilter.toLowerCase());
    }
    
    updateViolationsTable(filtered);
}

function viewViolation(id) {
    const violation = allViolations.find(v => v.id === id);
    if (violation) {
        alert(`Violation Details:\n\nID: ${violation.id}\nType: ${violation.violation_type}\nVehicle: ${violation.vehicle_class}\nSpeed: ${violation.speed_kmh} km/h\nTimestamp: ${violation.timestamp}`);
    }
}

// ==================== UTILITY FUNCTIONS ====================

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        if (typeof value === 'number') {
            animateValue(element, parseInt(element.textContent) || 0, value, 500);
        } else {
            element.textContent = value;
        }
    }
}

function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) {
        return 'Just now';
    } else if (diff < 3600000) {
        return `${Math.floor(diff / 60000)} min ago`;
    } else if (diff < 86400000) {
        return `${Math.floor(diff / 3600000)} hour ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function showNotification(type, message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Different styling based on type
    let backgroundColor, icon;
    if (type === 'success') {
        backgroundColor = 'var(--primary-green)';
        icon = '<i class="fas fa-check-circle"></i>';
    } else if (type === 'error') {
        backgroundColor = 'var(--primary-red)';
        icon = '<i class="fas fa-exclamation-circle"></i>';
    } else if (type === 'info') {
        backgroundColor = 'var(--primary-purple)';
        icon = '<i class="fas fa-info-circle"></i>';
    } else {
        backgroundColor = '#FFA500';
        icon = '<i class="fas fa-bell"></i>';
    }
    
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        min-width: 300px;
        max-width: 500px;
        padding: 1rem 1.5rem;
        background: ${backgroundColor};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 14px;
    `;
    
    notification.innerHTML = `
        ${icon}
        <span style="flex: 1;">${message}</span>
        <button onclick="this.parentElement.remove()" style="background: none; border: none; color: white; cursor: pointer; font-size: 18px;">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(notification);
    
    // Play sound for error (violation) notifications
    if (type === 'error') {
        playNotificationSound();
    }
    
    // Remove after 5 seconds (longer for violations)
    const duration = type === 'error' ? 5000 : 3000;
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

function playNotificationSound() {
    // Create audio context for notification beep
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    } catch (error) {
        console.log('Audio notification not available');
    }
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ==================== HEALTH CHECK ====================

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        console.log('System health:', data);
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Check health every 30 seconds
setInterval(checkHealth, 30000);

console.log('PTU Traffic Monitoring System initialized successfully!');
