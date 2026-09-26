import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function ViolationAlert({ violations = [] }) {
  if (!violations || violations.length === 0) {
    return null;
  }

  return (
    <div className="alert-banner">
      <div className="alert-content">
        <AlertTriangle size={20} color="var(--color-violation)" />
        <div>
          <strong>SAFETY VIOLATION ALERT: </strong>
          <span>
            {violations.length} worker{violations.length > 1 ? 's' : ''} detected without mandatory PPE:
          </span>
          <span style={{ marginLeft: '0.5rem', fontFamily: 'var(--font-mono)' }}>
            {violations.map((v) => `${v.employee_id !== 'UNKNOWN' ? v.employee_id : v.track_id} (Missing: ${v.missing_items.join(', ')})`).join(' | ')}
          </span>
        </div>
      </div>
    </div>
  );
}
