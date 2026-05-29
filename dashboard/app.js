/**
 * Store Intelligence - Live Dashboard Client
 * Connects via WebSocket for real-time metric updates.
 * Falls back to polling if WebSocket unavailable.
 */

const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}`;

let ws = null;
let currentStore = 'STORE_BLR_002';
let pollInterval = null;
let eventCount = 0;

// ─── WebSocket Connection ────────────────────────────────────────────────────

function connectToStore(storeId) {
    currentStore = storeId;
    
    // Close existing connection
    if (ws) {
        ws.close();
    }

    const wsUrl = `${WS_BASE}/ws/${storeId}`;
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = `Connected to ${storeId}`;
            // Fetch initial metrics
            fetchMetrics();
            fetchFunnel();
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleMessage(data);
        };
        
        ws.onclose = () => {
            document.getElementById('statusDot').classList.remove('connected');
            document.getElementById('statusText').textContent = 'Disconnected - retrying...';
            // Retry after 3s
            setTimeout(() => connectToStore(currentStore), 3000);
        };
        
        ws.onerror = () => {
            // Fall back to polling
            startPolling();
        };
    } catch (e) {
        startPolling();
    }
}

function startPolling() {
    document.getElementById('statusText').textContent = `Polling ${currentStore}`;
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(() => {
        fetchMetrics();
        fetchFunnel();
    }, 5000);
    // Initial fetch
    fetchMetrics();
    fetchFunnel();
}

// ─── Message Handling ────────────────────────────────────────────────────────

function handleMessage(data) {
    if (data.type === 'metrics_update') {
        updateMetrics(data.payload);
    } else if (data.type === 'event') {
        addEventToFeed(data);
        // Refresh metrics on each event
        fetchMetrics();
    }
}

// ─── API Calls ───────────────────────────────────────────────────────────────

async function fetchMetrics() {
    try {
        const resp = await fetch(`${API_BASE}/stores/${currentStore}/metrics`);
        if (resp.ok) {
            const data = await resp.json();
            updateMetrics(data);
        }
    } catch (e) {
        console.error('Failed to fetch metrics:', e);
    }
}

async function fetchFunnel() {
    try {
        const resp = await fetch(`${API_BASE}/stores/${currentStore}/funnel`);
        if (resp.ok) {
            const data = await resp.json();
            updateFunnel(data);
        }
    } catch (e) {
        console.error('Failed to fetch funnel:', e);
    }
}

// ─── UI Updates ──────────────────────────────────────────────────────────────

function updateMetrics(metrics) {
    animateValue('metric-visitors', metrics.unique_visitors);
    animateValue('metric-conversion', (metrics.conversion_rate * 100).toFixed(1) + '%');
    animateValue('metric-occupancy', metrics.current_occupancy);
    animateValue('metric-queue', metrics.current_queue_depth);
    animateValue('metric-abandonment', (metrics.abandonment_rate * 100).toFixed(1) + '%');
    
    // Average dwell across all zones
    const dwells = Object.values(metrics.avg_dwell_ms_per_zone || {});
    const avgDwell = dwells.length > 0 
        ? (dwells.reduce((a, b) => a + b, 0) / dwells.length / 1000).toFixed(1) + 's'
        : '--';
    animateValue('metric-avgdwell', avgDwell);
}

function updateFunnel(funnel) {
    const container = document.getElementById('funnel-bars');
    if (!funnel.stages || funnel.stages.length === 0) {
        container.innerHTML = '<div style="color: #71767b;">No data yet</div>';
        return;
    }
    
    const maxCount = funnel.stages[0].count || 1;
    
    container.innerHTML = funnel.stages.map(stage => `
        <div class="funnel-bar">
            <div class="stage-name">${stage.stage}</div>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${(stage.count / maxCount) * 100}%"></div>
            </div>
            <div class="count">${stage.count}</div>
        </div>
    `).join('');
}

function addEventToFeed(data) {
    const feed = document.getElementById('eventsFeed');
    eventCount++;
    
    // Keep max 50 events
    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }
    
    const typeClass = getEventTypeClass(data.event_type);
    const time = new Date(data.timestamp).toLocaleTimeString();
    
    const item = document.createElement('div');
    item.className = 'event-item';
    item.innerHTML = `
        <span class="event-type ${typeClass}">${data.event_type}</span>
        <span class="event-detail">${data.payload.visitor_id || ''} ${data.payload.zone_id ? '→ ' + data.payload.zone_id : ''}</span>
        <span class="event-time">${time}</span>
    `;
    
    feed.insertBefore(item, feed.firstChild);
}

function getEventTypeClass(type) {
    if (type === 'ENTRY' || type === 'REENTRY') return 'entry';
    if (type === 'EXIT') return 'exit';
    if (type.includes('DWELL')) return 'dwell';
    if (type.includes('BILLING')) return 'billing';
    return '';
}

function animateValue(elementId, newValue) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    const oldValue = el.textContent;
    el.textContent = newValue;
    
    if (oldValue !== String(newValue) && oldValue !== '--') {
        const card = el.closest('.metric-card');
        if (card) {
            card.classList.add('updated');
            setTimeout(() => card.classList.remove('updated'), 1000);
        }
    }
}

// ─── Initialize ──────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    connectToStore(currentStore);
});
