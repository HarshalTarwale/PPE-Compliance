import React, { useState } from 'react';
import { CheckCircle2, XCircle, UserX, UserCheck, ShieldCheck, ShieldAlert, Filter } from 'lucide-react';

export default function EmployeeTable({ persons = [] }) {
  const [filter, setFilter] = useState('ALL'); // 'ALL' | 'COMPLIANT' | 'VIOLATION'

  const filteredPersons = persons.filter((p) => {
    if (filter === 'COMPLIANT') return p.is_compliant;
    if (filter === 'VIOLATION') return !p.is_compliant;
    return true;
  });

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <ShieldCheck size={18} />
          <span>Real-Time Tracked Personnel ({persons.length})</span>
        </div>
        <div className="panel-controls">
          <div className="header-nav" style={{ padding: '0.15rem' }}>
            <button
              className={`nav-link ${filter === 'ALL' ? 'active' : ''}`}
              style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}
              onClick={() => setFilter('ALL')}
            >
              All ({persons.length})
            </button>
            <button
              className={`nav-link ${filter === 'COMPLIANT' ? 'active' : ''}`}
              style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}
              onClick={() => setFilter('COMPLIANT')}
            >
              Compliant ({persons.filter((p) => p.is_compliant).length})
            </button>
            <button
              className={`nav-link ${filter === 'VIOLATION' ? 'active' : ''}`}
              style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}
              onClick={() => setFilter('VIOLATION')}
            >
              Violations ({persons.filter((p) => !p.is_compliant).length})
            </button>
          </div>
        </div>
      </div>

      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Track ID</th>
              <th>Personnel Identity</th>
              <th>Hardhat</th>
              <th>Safety Vest</th>
              <th>Compliance</th>
              <th>Missing PPE</th>
              <th>Confidence</th>
              <th>Match Sim</th>
            </tr>
          </thead>
          <tbody>
            {filteredPersons.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                  No active personnel detected in current monitoring field.
                </td>
              </tr>
            ) : (
              filteredPersons.map((p) => (
                <tr key={p.track_id} className={!p.is_compliant ? 'row-violation' : ''}>
                  <td className="mono" style={{ fontWeight: 600 }}>
                    {p.track_id}
                  </td>
                  <td>
                    {p.employee_id !== 'UNKNOWN' ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <UserCheck size={14} color="var(--color-compliant)" />
                        <div>
                          <strong>{p.employee_name}</strong>
                          <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                            {p.employee_id}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <UserX size={14} color="var(--color-warning)" />
                        <span className="badge badge-warning">UNKNOWN</span>
                      </div>
                    )}
                  </td>
                  <td>
                    {p.hardhat ? (
                      <span className="badge badge-compliant">
                        <CheckCircle2 size={12} /> PRESENT
                      </span>
                    ) : (
                      <span className="badge badge-violation">
                        <XCircle size={12} /> MISSING
                      </span>
                    )}
                  </td>
                  <td>
                    {p.safety_vest ? (
                      <span className="badge badge-compliant">
                        <CheckCircle2 size={12} /> PRESENT
                      </span>
                    ) : (
                      <span className="badge badge-violation">
                        <XCircle size={12} /> MISSING
                      </span>
                    )}
                  </td>
                  <td>
                    {p.is_compliant ? (
                      <span className="badge badge-compliant">COMPLIANT</span>
                    ) : (
                      <span className="badge badge-violation">NON_COMPLIANT</span>
                    )}
                  </td>
                  <td>
                    {p.missing_items && p.missing_items.length > 0 ? (
                      <span className="mono" style={{ color: 'var(--color-violation)', fontSize: '0.75rem' }}>
                        {p.missing_items.join(', ')}
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>—</span>
                    )}
                  </td>
                  <td className="mono">
                    {(p.detection_confidence * 100).toFixed(1)}%
                  </td>
                  <td className="mono">
                    {p.face_similarity !== null && p.face_similarity !== undefined
                      ? (p.face_similarity * 100).toFixed(1) + '%'
                      : '—'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
