import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  X,
  ArrowLeft,
  ShieldCheck,
  UserPlus,
  RefreshCw,
  Radio,
  Sparkles,
} from 'lucide-react';
import { registerEmployee, fetchCameraSnapshot } from '../api/client';

export default function RegisterPage({ onNavigateEmployees, isMockMode }) {
  const [employeeId, setEmployeeId] = useState('');
  const [employeeName, setEmployeeName] = useState('');
  const [allowOverwrite, setAllowOverwrite] = useState(false);
  const [captureMode, setCaptureMode] = useState('camera'); // 'camera' | 'upload'
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [isFlashing, setIsFlashing] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Initialize and stop webcam stream based on active mode
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (e) {
          console.warn('Video play deferred:', e);
        }
      }
      setIsCameraActive(true);
      setCameraError(null);
    } catch (err) {
      console.warn('Browser webcam getUserMedia error:', err);
      setIsCameraActive(false);
      setCameraError(
        'Please allow browser camera permissions (click the lock/tune icon in your browser URL bar).'
      );
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  useEffect(() => {
    if (captureMode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
    return () => {
      stopCamera();
    };
  }, [captureMode]);

  // Take snapshot using browser webcam canvas
  const handleSnapWebcam = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Flip horizontally so the saved snapshot matches the mirrored user view
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height);

    // Visual shutter flash effect
    setIsFlashing(true);
    setTimeout(() => setIsFlashing(false), 120);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const filename = `snap_${Date.now()}_sample${previews.length + 1}.jpg`;
        const file = new File([blob], filename, { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);

        setSelectedFiles((prev) => [...prev, file]);
        setPreviews((prev) => [...prev, { name: filename, url, file }]);
      },
      'image/jpeg',
      0.95
    );
  };

  // Take snapshot directly from the backend camera feed (clean unannotated frame)
  const handleSnapSystemStream = async () => {
    setErrorMsg(null);
    setIsCapturing(true);

    // Visual shutter flash effect
    setIsFlashing(true);
    setTimeout(() => setIsFlashing(false), 120);

    try {
      const blob = await fetchCameraSnapshot(true);
      const filename = `camera_snap_${Date.now()}_sample${previews.length + 1}.jpg`;
      const file = new File([blob], filename, { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);

      setSelectedFiles((prev) => [...prev, file]);
      setPreviews((prev) => [...prev, { name: filename, url, file }]);
    } catch (err) {
      setErrorMsg(`Could not grab snapshot from camera stream: ${err.message}`);
    } finally {
      setIsCapturing(false);
    }
  };

  // Unified snapshot action: uses webcam if active, otherwise live stream snapshot
  const handleSnap = async () => {
    if (isCameraActive) {
      handleSnapWebcam();
    } else {
      await handleSnapSystemStream();
    }
  };

  // Upload file picker handler
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const newFiles = [...selectedFiles, ...files];
    setSelectedFiles(newFiles);

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

  const clearAllPhotos = () => {
    previews.forEach((p) => {
      if (p?.url) URL.revokeObjectURL(p.url);
    });
    setSelectedFiles([]);
    setPreviews([]);
  };

  // Revoke object URLs on unmount
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
      setErrorMsg('Please take or upload at least one face photo for enrollment.');
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
          message: `(Mock Mode) Successfully enrolled '${cleanName}' with ${selectedFiles.length} photo sample(s).`,
        });
        clearAllPhotos();
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
      setEmployeeId('');
      setEmployeeName('');
      clearAllPhotos();
    } catch (err) {
      setErrorMsg(err.message || 'Registration failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: '820px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Top Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2>Employee Biometric Enrollment</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem' }}>
            Take face photos directly with the camera to compute normalized facial recognition embeddings.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={onNavigateEmployees}>
          <ArrowLeft size={14} />
          <span>Back to Personnel</span>
        </button>
      </div>

      {/* Result feedback */}
      {result && result.success && (
        <div
          className="badge badge-compliant"
          style={{
            padding: '0.85rem 1.1rem',
            fontSize: '0.85rem',
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
          }}
        >
          <CheckCircle2 size={20} />
          <div>
            <strong>Registration Succeeded: </strong>
            <span>{result.message}</span>
            <div style={{ marginTop: '0.25rem', fontSize: '0.75rem', opacity: 0.9 }}>
              Valid Samples: {result.valid_samples} | Rejected: {result.rejected_samples}
            </div>
          </div>
        </div>
      )}

      {errorMsg && (
        <div
          className="badge badge-violation"
          style={{
            padding: '0.85rem 1.1rem',
            fontSize: '0.85rem',
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
          }}
        >
          <AlertCircle size={20} />
          <div>
            <strong>Notice: </strong>
            <span>{errorMsg}</span>
          </div>
        </div>
      )}

      {/* Form Container */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <UserPlus size={16} />
            <span>Personnel Enrollment Form</span>
          </div>
        </div>

        <div className="panel-body">
          <form onSubmit={handleSubmit} className="form-grid" style={{ maxWidth: '100%' }}>
            {/* Identity Inputs */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1rem' }}>
              <div className="form-group">
                <label className="form-label">Employee ID *</label>
                <input
                  type="text"
                  className="form-input mono"
                  placeholder="e.g. EMP102"
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
                  placeholder="e.g. John Doe"
                  value={employeeName}
                  onChange={(e) => setEmployeeName(e.target.value)}
                  disabled={isSubmitting}
                  required
                />
              </div>
            </div>

            <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="checkbox"
                id="allow-overwrite"
                checked={allowOverwrite}
                onChange={(e) => setAllowOverwrite(e.target.checked)}
                disabled={isSubmitting}
              />
              <label
                htmlFor="allow-overwrite"
                style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', cursor: 'pointer' }}
              >
                Allow updating existing employee record if ID matches
              </label>
            </div>

            {/* Photo Capture Section */}
            <div className="form-group">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <label className="form-label" style={{ margin: 0 }}>
                  Face Photos (Take 1-3 Angles) *
                </label>

                {/* Mode Switcher */}
                <div className="header-nav" style={{ padding: '0.15rem' }}>
                  <button
                    type="button"
                    className={`nav-link ${captureMode === 'camera' ? 'active' : ''}`}
                    onClick={() => setCaptureMode('camera')}
                    style={{ padding: '0.25rem 0.65rem', fontSize: '0.75rem' }}
                  >
                    <Camera size={14} />
                    <span>Take Photo</span>
                  </button>
                  <button
                    type="button"
                    className={`nav-link ${captureMode === 'upload' ? 'active' : ''}`}
                    onClick={() => setCaptureMode('upload')}
                    style={{ padding: '0.25rem 0.65rem', fontSize: '0.75rem' }}
                  >
                    <UploadCloud size={14} />
                    <span>Upload File</span>
                  </button>
                </div>
              </div>

              {/* CAMERA CAPTURE MODE */}
              {captureMode === 'camera' && (
                <div>
                  <div className="camera-viewfinder-box">
                    <video
                      ref={videoRef}
                      className="camera-video-element"
                      autoPlay
                      playsInline
                      muted
                      style={{
                        display: isCameraActive ? 'block' : 'none',
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                      }}
                    />
                    {!isCameraActive && (
                      <div
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          height: '100%',
                          padding: '1.5rem',
                          textAlign: 'center',
                          color: 'var(--text-muted)',
                          gap: '0.65rem',
                        }}
                      >
                        <Camera size={36} style={{ opacity: 0.7 }} />
                        <span style={{ fontSize: '0.875rem', fontWeight: 500 }}>
                          Webcam not active or permission pending
                        </span>
                        <button
                          type="button"
                          className="btn btn-primary btn-sm"
                          onClick={startCamera}
                          style={{ marginTop: '0.25rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                        >
                          <Camera size={14} />
                          <span>Enable Browser Camera</span>
                        </button>
                      </div>
                    )}
                    <div className="camera-face-guide" />
                    <div className="camera-guide-text">
                      Center face within oval &bull; Look straight &bull; Good lighting
                    </div>
                    <div className={`camera-shutter-flash ${isFlashing ? 'flashing' : ''}`} />
                  </div>

                  {cameraError && (
                    <div
                      style={{
                        fontSize: '0.75rem',
                        color: 'var(--color-telemetry)',
                        backgroundColor: 'var(--color-telemetry-bg)',
                        border: '1px solid var(--color-telemetry-border)',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm)',
                        marginTop: '0.5rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                      }}
                    >
                      <Sparkles size={14} />
                      <span>{cameraError}</span>
                    </div>
                  )}

                  {/* Camera Controls */}
                  <div className="capture-btn-group">
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button
                        type="button"
                        className="btn btn-primary"
                        onClick={handleSnap}
                        disabled={isSubmitting || isCapturing}
                        title="Capture photo from camera"
                      >
                        <Camera size={15} />
                        <span>{isCapturing ? 'Capturing...' : 'Take Photo (Snap)'}</span>
                      </button>

                      {isCameraActive && (
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={handleSnapSystemStream}
                          disabled={isSubmitting || isCapturing}
                          title="Grab unannotated photo from system stream"
                        >
                          <Radio size={14} />
                          <span>Stream Snap</span>
                        </button>
                      )}

                      {!isCameraActive && (
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={startCamera}
                          title="Try browser webcam"
                        >
                          <RefreshCw size={14} />
                          <span>Retry Browser Webcam</span>
                        </button>
                      )}
                    </div>

                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {previews.length === 0 ? (
                        <span style={{ color: 'var(--color-warning)' }}>No photo taken yet</span>
                      ) : (
                        <span style={{ color: 'var(--color-compliant)' }}>
                          ✓ {previews.length} photo(s) captured (2-3 angles recommended)
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* UPLOAD FILE MODE */}
              {captureMode === 'upload' && (
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
              )}

              {/* Photos Gallery (Shared for both webcam snapshots & uploads) */}
              {previews.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                      Enrolled Photo Samples ({previews.length}):
                    </span>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={clearAllPhotos}
                      style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem' }}
                    >
                      Clear All
                    </button>
                  </div>

                  <div className="preview-grid">
                    {previews.map((item, idx) => (
                      <div key={idx} className="preview-item">
                        <img src={item.url} alt={item.name} className="preview-img" />
                        <button
                          type="button"
                          className="preview-remove"
                          onClick={() => removeFile(idx)}
                          title="Remove this photo"
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
                style={{ padding: '0.65rem 1.35rem', fontSize: '0.85rem' }}
                disabled={isSubmitting || selectedFiles.length === 0}
              >
                <ShieldCheck size={16} />
                <span>
                  {isSubmitting
                    ? 'Computing Face Embeddings...'
                    : `Submit Enrollment (${selectedFiles.length} photo${selectedFiles.length === 1 ? '' : 's'})`}
                </span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

