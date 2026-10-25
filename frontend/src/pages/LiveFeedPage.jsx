import React from 'react';
import { Camera, Users, ShieldCheck, ShieldAlert, CheckCircle2, XCircle, UserCheck, UserX, Radio } from 'lucide-react';
import LiveMonitoring from '../components/LiveMonitoring';

export default function LiveFeedPage({
  monitoringStatus,
  onStartStream,
  onStopStream,
  isStreamLoading,
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

  return (
    <div className="live-page-layout">
      {/* Primary Video Feed Viewport */}
      <div className="live-viewport-col">
        <LiveMonitoring
          isActive={is_active}
          source={source}
          persons={persons}
          telemetry={telemetry}
          totalTracked={total_tracked}
          compliantCount={compliant_count}
          nonCompliantCount={non_compliant_count}
          onStart={onStartStream}
          onStop={onStopStream}
          isLoading={isStreamLoading}
        />
      </div>

      {/* Real-Time Personnel Detection Sidebar */}
      <div className="live-sidebar-col">
        <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div className="panel-header">
            <div className="panel-title">
              <Users size={18} />
              <span>Detections in FOV ({persons.length})</span>
            </div>
            <span
              className="badge"
              style={{
                backgroundColor: is_active ? 'var(--color-compliant-bg)' : 'var(--bg-surface)',
                color: is_active ? 'var(--color-compliant)' : 'var(--text-muted)',
                borderColor: is_active ? 'var(--color-compliant-border)' : 'var(--border-default)',
              }}
            >
              {is_active ? 'LIVE' : 'STANDBY'}
            </span>
          </div>

          <div className="panel-body" style={{ flex: 1, overflowY: 'auto', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            {persons.length === 0 ? (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  color: 'var(--text-muted)',
                  textAlign: 'center',
                  padding: '2rem 1rem',
                  gap: '0.5rem',
                }}
              >
                <Radio size={28} style={{ opacity: 0.4 }} />
                <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>No Personnel in View</div>
                <div style={{ fontSize: '0.75rem' }}>
                  {is_active
                    ? 'Scanning camera field of view in real time...'
                    : 'Start stream to begin live detection & verification.'}
                </div>
              </div>
            ) : (
              persons.map((person) => {
                const isCompliant = person.is_compliant;
                return (
                  <div
                    key={person.track_id}
                    style={{
                      backgroundColor: 'var(--bg-app)',
                      border: `1px solid ${isCompliant ? 'var(--color-compliant-border)' : 'var(--color-violation-border)'}`,
                      borderRadius: 'var(--radius-md)',
                      padding: '0.75rem',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.5rem',
                    }}
                  >
                    {/* Header: Track ID, Identity, Status */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        {person.employee_id !== 'UNKNOWN' ? (
                          <UserCheck size={16} color="var(--color-compliant)" />
                        ) : (
                          <UserX size={16} color="var(--color-warning)" />
                        )}
                        <div>
                          <div style={{ fontSize: '0.825rem', fontWeight: 700, color: 'var(--text-main)' }}>
                            {person.employee_id !== 'UNKNOWN' ? person.employee_name : 'UNKNOWN'}
                          </div>
                          <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                            {person.track_id} {person.employee_id !== 'UNKNOWN' ? `(${person.employee_id})` : ''}
                          </div>
                        </div>
                      </div>

                      <span
                        className={`badge ${isCompliant ? 'badge-compliant' : 'badge-violation'}`}
                        style={{ fontSize: '0.7rem' }}
                      >
                        {isCompliant ? (
                          <>
                            <ShieldCheck size={12} /> COMPLIANT
                          </>
                        ) : (
                          <>
                            <ShieldAlert size={12} /> VIOLATION
                          </>
                        )}
                      </span>
                    </div>

                    {/* Equipment badges */}
                    <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                      <span
                        className="badge"
                        style={{
                          fontSize: '0.7rem',
                          backgroundColor: person.hardhat ? 'var(--color-compliant-bg)' : 'var(--color-violation-bg)',
                          color: person.hardhat ? 'var(--color-compliant)' : 'var(--color-violation)',
                          borderColor: person.hardhat ? 'var(--color-compliant-border)' : 'var(--color-violation-border)',
                        }}
                      >
                        {person.hardhat ? <CheckCircle2 size={11} /> : <XCircle size={11} />} Hardhat
                      </span>

                      <span
                        className="badge"
                        style={{
                          fontSize: '0.7rem',
                          backgroundColor: person.safety_vest ? 'var(--color-compliant-bg)' : 'var(--color-violation-bg)',
                          color: person.safety_vest ? 'var(--color-compliant)' : 'var(--color-violation)',
                          borderColor: person.safety_vest ? 'var(--color-compliant-border)' : 'var(--color-violation-border)',
                        }}
                      >
                        {person.safety_vest ? <CheckCircle2 size={11} /> : <XCircle size={11} />} Safety Vest
                      </span>

                      {person.face_similarity !== null && person.face_similarity !== undefined && (
                        <span className="badge badge-neutral mono" style={{ fontSize: '0.68rem', marginLeft: 'auto' }}>
                          Sim: {(person.face_similarity * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
