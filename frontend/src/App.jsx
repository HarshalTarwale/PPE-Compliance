import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import UnknownPersonToasts from './components/UnknownPersonToasts';
import LiveFeedPage from './pages/LiveFeedPage';
import DashboardPage from './pages/DashboardPage';
import EmployeesPage from './pages/EmployeesPage';
import RegisterPage from './pages/RegisterPage';
import ViolationsPage from './pages/ViolationsPage';
import { MonitoringWebSocketManager } from './api/websocket';
import { fetchHealth, fetchMonitoringStatus, startMonitoring, stopMonitoring } from './api/client';
import { ISOLATED_DEV_MOCK_STATUS } from './api/mockData';
import './styles/index.css';

// How long an unknown-person notification stays on screen
const TOAST_MS = 8000;
// Ignore alerts older than this (e.g. replayed in the snapshot sent on reconnect)
const ALERT_MAX_AGE_MS = 60000;
// Free hosting (Render) sleeps after ~15 min without traffic; ping well before that
const KEEP_AWAKE_MS = 4 * 60 * 1000;

export default function App() {
  // Navigation: sync with pathname (/live, /dashboard, /employees, /register)
  const getTabFromPath = () => {
    const path = window.location.pathname.toLowerCase();
    if (path.includes('violation')) return 'violations';
    if (path.includes('live') || path.includes('stream') || path.includes('camera')) return 'live';
    if (path.includes('employee') || path.includes('personnel')) return 'employees';
    if (path.includes('register')) return 'register';
    return 'dashboard';
  };

  const [activeTab, setActiveTab] = useState(getTabFromPath);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [monitoringStatus, setMonitoringStatus] = useState(null);
  const [isMockMode, setIsMockMode] = useState(false);
  const [isStreamLoading, setIsStreamLoading] = useState(false);
  const [toasts, setToasts] = useState([]);

  const wsManagerRef = useRef(null);
  const seenEventIdsRef = useRef(new Set());

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Every status update (WebSocket or browser-camera frame) goes through here so
  // new unknown-person events pop up as notifications exactly once.
  const handleStatus = useCallback((status) => {
    setMonitoringStatus(status);
    const fresh = (status?.events || []).filter(
      (e) =>
        e.type === 'unknown_person' &&
        !seenEventIdsRef.current.has(e.id) &&
        Date.now() - Date.parse(e.timestamp) < ALERT_MAX_AGE_MS,
    );
    (status?.events || []).forEach((e) => seenEventIdsRef.current.add(e.id));
    if (fresh.length === 0) return;
    // People who appear together get one notification instead of a wall of them
    const toast = {
      id: fresh[0].id,
      timestamp: fresh[0].timestamp,
      trackIds: fresh.map((e) => e.track_id),
    };
    setToasts((prev) => [toast, ...prev].slice(0, 3));
    setTimeout(() => dismissToast(toast.id), TOAST_MS);
  }, [dismissToast]);

  // Wake the backend as soon as the page opens and keep it awake while it stays open
  useEffect(() => {
    if (isMockMode) return undefined;
    const ping = () => fetchHealth().catch(() => {});
    ping();
    const timer = setInterval(ping, KEEP_AWAKE_MS);
    return () => clearInterval(timer);
  }, [isMockMode]);

  // Sync route changes
  const navigateTo = (tab) => {
    setActiveTab(tab);
    const targetPath = tab === 'dashboard' ? '/dashboard' : `/${tab}`;
    window.history.pushState({}, '', targetPath);
  };

  useEffect(() => {
    const handlePopState = () => {
      setActiveTab(getTabFromPath());
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Connect WebSocket for live streaming
  useEffect(() => {
    if (isMockMode) {
      setMonitoringStatus(ISOLATED_DEV_MOCK_STATUS);
      setIsWsConnected(true);
      return;
    }

    // Initial fetch via REST
    fetchMonitoringStatus()
      .then((data) => setMonitoringStatus(data))
      .catch((err) => {
        console.warn('Initial REST status fetch error (backend may still be booting):', err.message);
      });

    // Establish persistent WebSocket connection
    const manager = new MonitoringWebSocketManager({
      onStatusUpdate: handleStatus,
      onConnectionChange: (connected) => {
        setIsWsConnected(connected);
      },
    });

    manager.connect();
    wsManagerRef.current = manager;

    return () => {
      manager.disconnect();
    };
  }, [isMockMode, handleStatus]);

  // Stream controls
  const handleStartStream = async (source = 0, flipHorizontal = true, flipVertical = false) => {
    setIsStreamLoading(true);
    if (isMockMode) {
      setTimeout(() => {
        setMonitoringStatus((prev) => ({
          ...(prev || ISOLATED_DEV_MOCK_STATUS),
          is_active: true,
          source: `Source #${source}`,
        }));
        setIsStreamLoading(false);
      }, 300);
      return;
    }

    try {
      await startMonitoring(source, flipHorizontal, flipVertical);
      const updated = await fetchMonitoringStatus();
      setMonitoringStatus(updated);
    } catch (err) {
      alert(`Could not start stream: ${err.message}`);
    } finally {
      setIsStreamLoading(false);
    }
  };

  const handleStopStream = async () => {
    setIsStreamLoading(true);
    if (isMockMode) {
      setTimeout(() => {
        setMonitoringStatus((prev) => ({
          ...(prev || ISOLATED_DEV_MOCK_STATUS),
          is_active: false,
          persons: [],
          total_tracked: 0,
          compliant_count: 0,
          non_compliant_count: 0,
        }));
        setIsStreamLoading(false);
      }, 300);
      return;
    }

    try {
      await stopMonitoring();
      const updated = await fetchMonitoringStatus();
      setMonitoringStatus(updated);
    } catch (err) {
      alert(`Could not stop stream: ${err.message}`);
    } finally {
      setIsStreamLoading(false);
    }
  };

  const handleToggleMock = () => {
    setIsMockMode((prev) => !prev);
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        onTabChange={navigateTo}
        isWsConnected={isWsConnected}
        isStreamActive={monitoringStatus?.is_active || false}
        fps={monitoringStatus?.telemetry?.fps || 0}
        isMockMode={isMockMode}
        onToggleMock={handleToggleMock}
      />

      <main className="main-content">
        {activeTab === 'live' && (
          <LiveFeedPage
            monitoringStatus={monitoringStatus}
            onStartStream={handleStartStream}
            onStopStream={handleStopStream}
            onBrowserStatus={handleStatus}
            isStreamLoading={isStreamLoading}
          />
        )}

        {activeTab === 'dashboard' && (
          <DashboardPage
            monitoringStatus={monitoringStatus}
            onNavigateLive={() => navigateTo('live')}
          />
        )}

        {activeTab === 'employees' && (
          <EmployeesPage
            onNavigateRegister={() => navigateTo('register')}
            isMockMode={isMockMode}
          />
        )}

        {activeTab === 'violations' && <ViolationsPage isMockMode={isMockMode} />}

        {activeTab === 'register' && (
          <RegisterPage
            onNavigateEmployees={() => navigateTo('employees')}
            isMockMode={isMockMode}
          />
        )}
      </main>

      <UnknownPersonToasts toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
