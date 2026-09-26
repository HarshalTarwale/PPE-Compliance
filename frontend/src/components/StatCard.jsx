import React from 'react';

export default function StatCard({ label, value, sub, accent = 'neutral', icon: Icon }) {
  return (
    <div className={`metric-card accent-${accent}`}>
      <div className="metric-label">
        <span>{label}</span>
        {Icon && <Icon size={16} className="text-secondary" />}
      </div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}
