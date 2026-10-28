import React, { useState, useEffect } from 'react';
import { ShieldAlert, Activity, Users, UserPlus, Radio, Wifi, WifiOff, Video, History } from 'lucide-react';

export default function Header({
  activeTab,
  onTabChange,
  isWsConnected,
  isStreamActive,
  fps = 0,
  isMockMode,
  onToggleMock,
}) {
  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-icon">
          <ShieldAlert size={20} />
        </div>
        <div>
          <div className="brand-title">PPE Safety Monitoring &amp; Compliance</div>
          <div className="brand-subtitle">Automated Real-Time Visual Inspection System</div>
        </div>
      </div>

      <nav className="header-nav">
        <button
          className={`nav-link ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => onTabChange('live')}
          title="Live Camera & Sensor Stream"
        >
          <Video size={16} />
          <span>Live Feed</span>
          {isStreamActive && <span className="live-tab-badge" title="Live stream active" />}
        </button>
        <button
          className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => onTabChange('dashboard')}
        >
          <Activity size={16} />
          <span>Dashboard</span>
        </button>
        <button
          className={`nav-link ${activeTab === 'violations' ? 'active' : ''}`}
          onClick={() => onTabChange('violations')}
          title="History of PPE violations and unknown persons"
        >
          <History size={16} />
          <span>Violation Log</span>
        </button>
        <button
          className={`nav-link ${activeTab === 'employees' ? 'active' : ''}`}
          onClick={() => onTabChange('employees')}
        >
          <Users size={16} />
          <span>Personnel</span>
        </button>
        <button
          className={`nav-link ${activeTab === 'register' ? 'active' : ''}`}
          onClick={() => onTabChange('register')}
        >
          <UserPlus size={16} />
          <span>Register</span>
        </button>
      </nav>

      <div className="header-status">
        <div className="status-pill">
          <span className={`status-dot ${isWsConnected ? 'active' : 'inactive'}`} />
          <span>{isWsConnected ? 'WS LIVE' : 'WS OFFLINE'}</span>
        </div>

        <div className="status-pill">
          <Radio size={12} className={isStreamActive ? 'color-telemetry' : ''} />
          <span>{isStreamActive ? 'STREAM ACTIVE' : 'STREAM IDLE'}</span>
        </div>

        {isStreamActive && (
          <div className="status-pill mono">
            <span>FPS: {fps > 0 ? fps.toFixed(1) : '0.0'}</span>
          </div>
        )}

        <div className="status-pill mono" style={{ fontSize: '0.7rem' }}>
          <span>{utcTime}</span>
        </div>

        {/* Development mode indicator / toggle */}
        <button
          className="btn btn-secondary btn-sm mono"
          onClick={onToggleMock}
          title="Toggle isolated mock data for visual inspection without backend"
          style={{ fontSize: '0.68rem', padding: '0.2rem 0.5rem' }}
        >
          {isMockMode ? 'MOCK: ON' : 'BACKEND: LIVE'}
        </button>
      </div>
    </header>
  );
}
