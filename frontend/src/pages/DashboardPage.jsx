import React from 'react';
import { Users, ShieldCheck, ShieldAlert, Activity, Camera, ArrowRight, Radio } from 'lucide-react';
import StatCard from '../components/StatCard';
import EmployeeTable from '../components/EmployeeTable';
import ViolationAlert from '../components/ViolationAlert';

export default function DashboardPage({
  monitoringStatus,
  onNavigateLive,
}) {
  const {
    is_active = false,
    source = null,
    total_tracked = 0,
    compliant_count = 0,
    non_compliant_count = 0,
    persons = [],
    telemetry = {},
  } = monitoringStatus || {};

  const violations = persons.filter((p) => !p.is_compliant);
  const complianceRate =
    total_tracked > 0 ? ((compliant_count / total_tracked) * 100).toFixed(0) : 100;

  return (
    <div className="dashboard-page">
      {/* Safety Violation Alert Bar */}
      <ViolationAlert violations={violations} />

      {/* KPI Metric Readouts */}
      <div className="metrics-grid">
        <StatCard
          label="People Detected"
          value={total_tracked}
          sub={is_active ? 'Active tracking in FOV' : 'Monitoring standby'}
          accent="telemetry"
          icon={Users}
        />
        <StatCard
          label="Compliant Personnel"
          value={compliant_count}
          sub={`${complianceRate}% compliance rate`}
          accent="compliant"
          icon={ShieldCheck}
        />
        <StatCard
          label="Non-Compliant Violations"
          value={non_compliant_count}
          sub={non_compliant_count > 0 ? 'Immediate action required' : 'Zero violations'}
          accent={non_compliant_count > 0 ? 'violation' : 'neutral'}
          icon={ShieldAlert}
        />
        <StatCard
          label="Inference Telemetry"
          value={`${telemetry.fps ? telemetry.fps.toFixed(1) : '0.0'} FPS`}
          sub={`Total: ${telemetry.total_ms ? telemetry.total_ms.toFixed(1) : '0.0'} ms | Det: ${telemetry.detection_ms ? telemetry.detection_ms.toFixed(1) : '0.0'} ms`}
          accent="telemetry"
          icon={Activity}
        />
      </div>

      {/* Quick Camera Navigation Action Banner */}
      <div
        className="panel"
        style={{
          marginBottom: '1.25rem',
          padding: '0.85rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          background: is_active
            ? 'linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.95) 100%)'
            : 'var(--bg-panel)',
          borderLeft: `3px solid ${is_active ? 'var(--color-compliant)' : 'var(--border-strong)'}`,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <div
            style={{
              width: '36px',
              height: '36px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: is_active ? 'var(--color-compliant-bg)' : 'var(--bg-surface)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: is_active ? 'var(--color-compliant)' : 'var(--text-muted)',
            }}
          >
            <Camera size={18} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-main)' }}>
                Live Camera Sensor Feed
              </span>
              <span
                className="badge"
                style={{
                  fontSize: '0.68rem',
                  backgroundColor: is_active ? 'var(--color-compliant-bg)' : 'var(--bg-surface)',
                  color: is_active ? 'var(--color-compliant)' : 'var(--text-muted)',
                  borderColor: is_active ? 'var(--color-compliant-border)' : 'var(--border-default)',
                }}
              >
                {is_active ? 'STREAMING ACTIVE' : 'STANDBY'}
              </span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
              {is_active
                ? `Currently processing source ${source || '0'} at ${telemetry.fps ? telemetry.fps.toFixed(1) : '30'} FPS (${total_tracked} worker(s) in view)`
                : 'Sensor feed is on standby. Switch to Live Feed to start camera and view real-time overlays.'}
            </div>
          </div>
        </div>

        <button
          className="btn btn-primary btn-sm"
          onClick={onNavigateLive}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 1rem' }}
        >
          <span>{is_active ? 'View Live Video Stream' : 'Open Live Camera Feed'}</span>
          <ArrowRight size={14} />
        </button>
      </div>

      {/* Main Real-Time Tracked Personnel Status Table (Full Screen Fit) */}
      <EmployeeTable persons={persons} />
    </div>
  );
}
