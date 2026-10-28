import React from 'react';
import { UserX, X } from 'lucide-react';

/** Stack of orange "Unknown person detected" notifications (top-right corner). */
export default function UnknownPersonToasts({ toasts = [], onDismiss }) {
  if (toasts.length === 0) return null;

  return (
    <div className="toast-stack" role="status" aria-live="assertive">
      {toasts.map((t) => (
        <div key={t.id} className="toast toast-unknown">
          <div className="toast-icon">
            <UserX size={20} />
          </div>
          <div className="toast-body">
            <div className="toast-title">
              {t.trackIds.length > 1
                ? `${t.trackIds.length} Unknown Persons Detected`
                : 'Unknown Person Detected'}
            </div>
            <div className="toast-text">
              {t.trackIds.length > 1
                ? `Unregistered people (${t.trackIds.join(', ')}) are in the camera view.`
                : `An unregistered person (${t.trackIds[0]}) is in the camera view.`}
            </div>
            <div className="toast-time mono">{new Date(t.timestamp).toLocaleTimeString()}</div>
          </div>
          <button className="toast-close" onClick={() => onDismiss(t.id)} title="Dismiss">
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
