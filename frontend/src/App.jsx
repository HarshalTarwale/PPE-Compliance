import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import DashboardPage from './pages/DashboardPage';
import EmployeesPage from './pages/EmployeesPage';
import RegisterPage from './pages/RegisterPage';
import { MonitoringWebSocketManager } from './api/websocket';
import { fetchMonitoringStatus, startMonitoring, stopMonitoring } from './api/client';
import { ISOLATED_DEV_MOCK_STATUS } from './api/mockData';
import './styles/index.css';

export default function App() {
  // Navigation: sync with pathname (/dashboard, /employees, /register)
  const getTabFromPath = () => {
    const path = window.location.pathname.toLowerCase();
    if (path.includes('employee')) return 'employees';
    if (path.includes('register')) return 'register';
    return 'dashboard';
  };

  const [activeTab, setActiveTab] = useState(getTabFromPath);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [monitoringStatus, setMonitoringStatus] = useState(null);
  const [isMockMode, setIsMockMode] = useState(false);
  const [isStreamLoading, setIsStreamLoading] = useState(false);

  const wsManagerRef = useRef(null);

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
      onStatusUpdate: (status) => {
        setMonitoringStatus(status);
      },
      onConnectionChange: (connected) => {
        setIsWsConnected(connected);
      },
    });

    manager.connect();
    wsManagerRef.current = manager;

    return () => {
      manager.disconnect();
    };
  }, [isMockMode]);

  // Stream controls
  const handleStartStream = async (source = 0) => {
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
      await startMonitoring(source);
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
        {activeTab === 'dashboard' && (
          <DashboardPage
            monitoringStatus={monitoringStatus}
            onStartStream={handleStartStream}
            onStopStream={handleStopStream}
            isStreamLoading={isStreamLoading}
          />
        )}

        {activeTab === 'employees' && (
          <EmployeesPage
            onNavigateRegister={() => navigateTo('register')}
            isMockMode={isMockMode}
          />
        )}

        {activeTab === 'register' && (
          <RegisterPage
            onNavigateEmployees={() => navigateTo('employees')}
            isMockMode={isMockMode}
          />
        )}
      </main>
    </div>
  );
}
