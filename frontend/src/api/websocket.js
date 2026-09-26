/**
 * WebSocket manager for real-time PPE monitoring data streaming.
 * Connects to ws://localhost:8000/ws/monitoring.
 */

export class MonitoringWebSocketManager {
  constructor({ onStatusUpdate, onConnectionChange }) {
    this.onStatusUpdate = onStatusUpdate;
    this.onConnectionChange = onConnectionChange;
    this.ws = null;
    this.reconnectTimer = null;
    this.isExplicitlyClosed = false;
    this.reconnectAttempts = 0;
  }

  connect() {
    this.isExplicitlyClosed = false;
    clearTimeout(this.reconnectTimer);

    // Determine WebSocket URL from current browser location or configured env
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = import.meta.env.VITE_WS_URL || `${protocol}//${host}/ws/monitoring`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        if (this.onConnectionChange) {
          this.onConnectionChange(true);
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === 'snapshot' || payload.type === 'monitoring_update') {
            if (this.onStatusUpdate && payload.data) {
              this.onStatusUpdate(payload.data);
            }
          }
        } catch (err) {
          console.error('Failed to parse monitoring WebSocket message:', err);
        }
      };

      this.ws.onclose = () => {
        if (this.onConnectionChange) {
          this.onConnectionChange(false);
        }
        if (!this.isExplicitlyClosed) {
          this._scheduleReconnect();
        }
      };

      this.ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
      };
    } catch (err) {
      console.error('WebSocket initialization failure:', err);
      this._scheduleReconnect();
    }
  }

  _scheduleReconnect() {
    this.reconnectAttempts += 1;
    // Exponential backoff capped at 5 seconds
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 5000);
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  disconnect() {
    this.isExplicitlyClosed = true;
    clearTimeout(this.reconnectTimer);
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
