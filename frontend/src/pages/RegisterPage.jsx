import React, { useState, useEffect } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, X, ArrowLeft, ShieldCheck, UserPlus } from 'lucide-react';
import { registerEmployee } from '../api/client';

export default function RegisterPage({ onNavigateEmployees, isMockMode }) {
  const [employeeId, setEmployeeId] = useState('');
  const [employeeName, setEmployeeName] = useState('');
  const [allowOverwrite, setAllowOverwrite] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const newFiles = [...selectedFiles, ...files];
    setSelectedFiles(newFiles);

    // Generate local preview URLs
    const newPreviews = files.map((file) => ({
      name: file.name,
      url: URL.createObjectURL(file),
      file,
    }));
    setPreviews((prev) => [...prev, ...newPreviews]);
  };

  const removeFile = (indexToRemove) => {
    setSelectedFiles((prev) => prev.filter((_, idx) => idx !== indexToRemove));
    setPreviews((prev) => {
      const removed = prev[indexToRemove];
      if (removed) URL.revokeObjectURL(removed.url);
      return prev.filter((_, idx) => idx !== indexToRemove);
    });
  };

  // Revoke object URLs on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      previews.forEach((p) => {
        if (p?.url) URL.revokeObjectURL(p.url);
      });
    };
  }, [previews]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg(null);
    setResult(null);

    const cleanId = employeeId.trim();
    const cleanName = employeeName.trim();

    if (!cleanId) {
      setErrorMsg('Please enter a valid Employee ID.');
      return;
    }

    if (!cleanName) {
      setErrorMsg('Please enter a valid Employee Name.');
      return;
    }

    if (selectedFiles.length === 0) {
      setErrorMsg('Please select at least one face photo for enrollment.');
      return;
    }

    setIsSubmitting(true);

    if (isMockMode) {
      setTimeout(() => {
        setIsSubmitting(false);
        setResult({
          success: true,
          employee_id: cleanId,
          employee_name: cleanName,
          valid_samples: selectedFiles.length,
          rejected_samples: 0,
          message: `(Mock Mode) Successfully enrolled '${cleanName}' with ${selectedFiles.length} face sample(s).`,
        });
      }, 600);
      return;
    }

    const formData = new FormData();
    formData.append('employee_id', cleanId);
    formData.append('employee_name', cleanName);
    formData.append('allow_overwrite', allowOverwrite ? 'true' : 'false');

    selectedFiles.forEach((file) => {
      formData.append('images', file);
    });

    try {
      const resp = await registerEmployee(formData);
      setResult(resp);
      // Clear inputs on success
      setEmployeeId('');
      setEmployeeName('');
      previews.forEach((p) => {
        if (p?.url) URL.revokeObjectURL(p.url);
      });
      setPreviews([]);
    } catch (err) {
      setErrorMsg(err.message || 'Registration failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Top Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2>Employee Biometric Enrollment</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem' }}>
            Register new site personnel by extracting normalized face embeddings from multiple photos.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={onNavigateEmployees}>
          <ArrowLeft size={14} />
          <span>Back to Personnel</span>
        </button>
      </div>

      {/* Result feedback */}
      {result && result.success && (
        <div className="badge badge-compliant" style={{ padding: '0.75rem 1rem', fontSize: '0.85rem', width: '100%', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <CheckCircle2 size={18} />
          <div>
            <strong>Registration Succeeded: </strong>
            <span>{result.message}</span>
            <div style={{ marginTop: '0.25rem', fontSize: '0.75rem' }}>
              Valid Samples: {result.valid_samples} | Rejected: {result.rejected_samples}
            </div>
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="badge badge-violation" style={{ padding: '0.75rem 1rem', fontSize: '0.85rem', width: '100%', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={18} />
          <div>
            <strong>Error: </strong>
            <span>{errorMsg}</span>
          </div>
        </div>
      )}

      {/* Form Container */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <UserPlus size={16} />
            <span>Enrollment Specifications</span>
          </div>
        </div>

        <div className="panel-body">
          <form onSubmit={handleSubmit} className="form-grid">
            <div className="form-group">
              <label className="form-label">Employee ID *</label>
              <input
                type="text"
                className="form-input mono"
                placeholder="e.g. EMP104"
                value={employeeId}
                onChange={(e) => setEmployeeId(e.target.value)}
                disabled={isSubmitting}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Full Name *</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Dr. Robert Taylor"
                value={employeeName}
                onChange={(e) => setEmployeeName(e.target.value)}
                disabled={isSubmitting}
                required
              />
            </div>

            <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="checkbox"
                id="allow-overwrite"
                checked={allowOverwrite}
                onChange={(e) => setAllowOverwrite(e.target.checked)}
                disabled={isSubmitting}
              />
              <label htmlFor="allow-overwrite" style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                Allow updating existing employee record if ID matches
              </label>
            </div>

            {/* Photo Upload Zone */}
            <div className="form-group">
              <label className="form-label">Face Images (Multiple Angles Recommended) *</label>
              <label className="dropzone">
                <input
                  type="file"
                  multiple
                  accept="image/png, image/jpeg, image/jpg"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                  disabled={isSubmitting}
                />
                <UploadCloud size={32} color="var(--color-telemetry)" style={{ margin: '0 auto 0.5rem' }} />
                <div style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.9rem' }}>
                  Click to browse or drop images here
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Requirements: Clear lighting, single face per photo, PNG or JPEG.
                </div>
              </label>

              {/* Thumbnails */}
              {previews.length > 0 && (
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                    Selected Images ({previews.length}):
                  </div>
                  <div className="preview-grid">
                    {previews.map((item, idx) => (
                      <div key={idx} className="preview-item">
                        <img src={item.url} alt={item.name} className="preview-img" />
                        <button
                          type="button"
                          className="preview-remove"
                          onClick={() => removeFile(idx)}
                          title="Remove sample"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
              <button
                type="submit"
                className="btn btn-primary"
                style={{ padding: '0.65rem 1.25rem' }}
                disabled={isSubmitting}
              >
                <ShieldCheck size={16} />
                <span>{isSubmitting ? 'Computing Embeddings...' : 'Submit Enrollment'}</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
