import React, { useState, useEffect } from 'react';
import { Users, UserPlus, Trash2, Search, RefreshCw, Key, Image } from 'lucide-react';
import { fetchEmployees, deleteEmployee } from '../api/client';
import { ISOLATED_DEV_MOCK_EMPLOYEES } from '../api/mockData';

export default function EmployeesPage({ onNavigateRegister, isMockMode }) {
  const [employees, setEmployees] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [errorMsg, setErrorMsg] = useState(null);
  const [actionMsg, setActionMsg] = useState(null);

  const loadData = async () => {
    setIsLoading(true);
    setErrorMsg(null);

    if (isMockMode) {
      setEmployees(ISOLATED_DEV_MOCK_EMPLOYEES);
      setIsLoading(false);
      return;
    }

    try {
      const data = await fetchEmployees();
      setEmployees(data.employees || []);
    } catch (err) {
      console.warn('Backend unavailable, falling back to isolated mock list:', err);
      setErrorMsg(`Backend notice: ${err.message}. Showing mock data.`);
      setEmployees(ISOLATED_DEV_MOCK_EMPLOYEES);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [isMockMode]);

  const handleDelete = async (employeeId) => {
    if (!window.confirm(`Are you sure you want to delete employee record '${employeeId}'?`)) {
      return;
    }

    if (isMockMode) {
      setEmployees((prev) => prev.filter((e) => e.employee_id !== employeeId));
      setActionMsg(`Employee '${employeeId}' deleted in mock mode.`);
      return;
    }

    try {
      await deleteEmployee(employeeId);
      setActionMsg(`Employee '${employeeId}' deleted successfully.`);
      loadData();
    } catch (err) {
      setErrorMsg(`Failed to delete employee: ${err.message}`);
    }
  };

  const filteredEmployees = employees.filter((emp) => {
    const q = searchQuery.toLowerCase();
    return (
      emp.employee_id.toLowerCase().includes(q) ||
      emp.employee_name.toLowerCase().includes(q)
    );
  });

  const totalSamples = employees.reduce((acc, e) => acc + (e.sample_count || 0), 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Top Banner & Action Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2>Registered Personnel Catalog</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem' }}>
            Biometric face embeddings stored securely using OpenCV SFace 128-D vectors (No raw images retained).
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button className="btn btn-secondary btn-sm" onClick={loadData} disabled={isLoading}>
            <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
            <span>Refresh</span>
          </button>
          <button className="btn btn-primary btn-sm" onClick={onNavigateRegister}>
            <UserPlus size={14} />
            <span>Enroll New Employee</span>
          </button>
        </div>
      </div>

      {actionMsg && (
        <div className="badge badge-compliant" style={{ padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
          {actionMsg}
        </div>
      )}

      {errorMsg && (
        <div className="badge badge-warning" style={{ padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
          {errorMsg}
        </div>
      )}

      {/* Metrics Row */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
        <div className="metric-card accent-telemetry">
          <div className="metric-label">Enrolled Employees</div>
          <div className="metric-value">{employees.length}</div>
          <div className="metric-sub">Active identity records</div>
        </div>
        <div className="metric-card accent-compliant">
          <div className="metric-label">Stored Face Embeddings</div>
          <div className="metric-value">{totalSamples}</div>
          <div className="metric-sub">Normalized unit vectors</div>
        </div>
        <div className="metric-card accent-warning">
          <div className="metric-label">Embedding Dimensions</div>
          <div className="metric-value">128-D</div>
          <div className="metric-sub">SFace ONNX model</div>
        </div>
      </div>

      {/* Search Bar & Table */}
      <div className="panel">
        <div className="panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%', maxWidth: '380px' }}>
            <Search size={16} color="var(--text-muted)" />
            <input
              type="text"
              className="form-input"
              style={{ width: '100%', padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
              placeholder="Filter by ID or Name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Showing {filteredEmployees.length} of {employees.length}
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Employee ID</th>
                <th>Full Name</th>
                <th>Enrolled Date</th>
                <th>Face Samples</th>
                <th>Vector Dim</th>
                <th>Department / Role</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredEmployees.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-muted)' }}>
                    {isLoading ? 'Loading employee records...' : 'No enrolled employee records found.'}
                  </td>
                </tr>
              ) : (
                filteredEmployees.map((emp) => (
                  <tr key={emp.employee_id}>
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--color-telemetry)' }}>
                      {emp.employee_id}
                    </td>
                    <td>
                      <strong>{emp.employee_name}</strong>
                    </td>
                    <td className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {emp.registered_at ? new Date(emp.registered_at).toLocaleDateString() : '—'}
                    </td>
                    <td>
                      <span className="badge badge-neutral">
                        <Image size={11} style={{ marginRight: '3px' }} />
                        {emp.sample_count} sample{emp.sample_count > 1 ? 's' : ''}
                      </span>
                    </td>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>
                      {emp.embedding_dim || 128}-D
                    </td>
                    <td style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                      {emp.metadata?.role || emp.metadata?.department || 'Staff'}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-secondary btn-sm"
                        style={{ color: 'var(--color-violation)' }}
                        onClick={() => handleDelete(emp.employee_id)}
                        title="Delete employee record"
                      >
                        <Trash2 size={13} />
                        <span>Delete</span>
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
