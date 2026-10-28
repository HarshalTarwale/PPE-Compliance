import React, { useState, useEffect, useCallback } from 'react';
import { History, RefreshCw, Trash2, ShieldAlert, UserX, UserCheck, HardHat, Shirt } from 'lucide-react';
import { fetchSafetyEvents, clearSafetyEvents } from '../api/client';

const REFRESH_MS = 5000;

function formatTime(iso) {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' }),
    time: d.toLocaleTimeString(),
  };
}

function MissingBadges({ items }) {
  if (!items || items.length === 0) return <span style={{ color: 'var(--text-muted)' }}>—</span>;
  return (
    <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
      {items.map((item) => (
        <span key={item} className="badge badge-violation">
          {item.toLowerCase().includes('hat') ? <HardHat size={12} /> : <Shirt size={12} />}
          {item}
        </span>
      ))}
    </div>
  );
}

export default function ViolationsPage({ isMockMode }) {
  const [events, setEvents] = useState([]);
  const [total, setTotal] = useState(0);
  const [filter, setFilter] = useState('ALL'); // 'ALL' | 'ppe_violation' | 'unknown_person'
  const [isLoading, setIsLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState(null);

  const loadEvents = useCallback(async () => {
    if (isMockMode) {
      setEvents([]);
      setTotal(0);
      setIsLoading(false);
      return;
    }
    try {
      const data = await fetchSafetyEvents(500);
      setEvents(data.events || []);
      setTotal(data.total || 0);
      setErrorMsg(null);
    } catch (err) {
      setErrorMsg(`Could not load the violation log: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [isMockMode]);

  // Keep the log current while the page is open
  useEffect(() => {
    loadEvents();
    const timer = setInterval(loadEvents, REFRESH_MS);
    return () => clearInterval(timer);
  }, [loadEvents]);

  const handleClear = async () => {
    if (!window.confirm('Delete the entire violation history? This cannot be undone.')) return;
    try {
      await clearSafetyEvents();
      loadEvents();
    } catch (err) {
      setErrorMsg(`Could not clear the log: ${err.message}`);
    }
  };

  const violationCount = events.filter((e) => e.type === 'ppe_violation').length;
  const unknownCount = events.filter((e) => e.type === 'unknown_person').length;
  const hardhatCount = events.filter(
    (e) => e.type === 'ppe_violation' && e.missing_items.some((m) => m.toLowerCase().includes('hat')),
  ).length;
  const shown = filter === 'ALL' ? events : events.filter((e) => e.type === filter);

  const filterBtn = (key, label) => (
    <button
      className={`nav-link ${filter === key ? 'active' : ''}`}
      style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}
      onClick={() => setFilter(key)}
    >
      {label}
    </button>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2>Safety Violation Log</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem' }}>
            Every PPE violation and unknown person seen by the camera, with the exact time it happened.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button className="btn btn-secondary btn-sm" onClick={loadEvents} disabled={isLoading}>
            <RefreshCw size={14} />
            <span>Refresh</span>
          </button>
          <button className="btn btn-danger btn-sm" onClick={handleClear} disabled={events.length === 0}>
            <Trash2 size={14} />
            <span>Clear Log</span>
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="badge badge-warning" style={{ padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
          {errorMsg}
        </div>
      )}

      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
        <div className="metric-card accent-violation">
          <div className="metric-label">PPE Violations</div>
          <div className="metric-value">{violationCount}</div>
          <div className="metric-sub">Logged incidents</div>
        </div>
        <div className="metric-card accent-violation">
          <div className="metric-label">Hardhat Missing</div>
          <div className="metric-value">{hardhatCount}</div>
          <div className="metric-sub">Incidents without a hardhat</div>
        </div>
        <div className="metric-card accent-warning">
          <div className="metric-label">Unknown Persons</div>
          <div className="metric-value">{unknownCount}</div>
          <div className="metric-sub">Unregistered people detected</div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <History size={18} />
            <span>Event History ({total})</span>
          </div>
          <div className="panel-controls">
            <div className="header-nav" style={{ padding: '0.15rem' }}>
              {filterBtn('ALL', `All (${events.length})`)}
              {filterBtn('ppe_violation', `PPE Violations (${violationCount})`)}
              {filterBtn('unknown_person', `Unknown Persons (${unknownCount})`)}
            </div>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Time</th>
                <th>Event</th>
                <th>Person</th>
                <th>Missing PPE</th>
                <th>Track ID</th>
              </tr>
            </thead>
            <tbody>
              {shown.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    {isLoading
                      ? 'Loading…'
                      : 'No events logged yet. Start the stream on the Live Feed page; violations appear here automatically.'}
                  </td>
                </tr>
              ) : (
                shown.map((e) => {
                  const { date, time } = formatTime(e.timestamp);
                  const isUnknownEvent = e.type === 'unknown_person';
                  const known = e.employee_id !== 'UNKNOWN';
                  return (
                    <tr key={e.id} className={isUnknownEvent ? '' : 'row-violation'}>
                      <td className="mono">{date}</td>
                      <td className="mono" style={{ fontWeight: 600 }}>{time}</td>
                      <td>
                        {isUnknownEvent ? (
                          <span className="badge badge-warning">
                            <UserX size={12} /> UNKNOWN PERSON
                          </span>
                        ) : (
                          <span className="badge badge-violation">
                            <ShieldAlert size={12} /> PPE VIOLATION
                          </span>
                        )}
                      </td>
                      <td>
                        {known ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <UserCheck size={14} color="var(--color-compliant)" />
                            <div>
                              <strong>{e.employee_name}</strong>
                              <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                                {e.employee_id}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <span className="badge badge-warning">UNKNOWN</span>
                        )}
                      </td>
                      <td>
                        <MissingBadges items={e.missing_items} />
                      </td>
                      <td className="mono">{e.track_id}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
