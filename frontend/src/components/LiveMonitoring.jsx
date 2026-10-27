import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Play, Square, Camera, Video, FlipHorizontal, Laptop, Server } from 'lucide-react';
import { toggleCameraFlip, analyzeFrame, stopMonitoring, fetchMonitoringStatus } from '../api/client';

// Width of frames sent to the backend (height follows the camera aspect ratio)
const CAPTURE_WIDTH = 640;

// When the app is opened from anywhere other than this machine, the backend
// (e.g. on Render) has no webcam, so default to the viewer's own camera.
const isLocalHost = ['localhost', '127.0.0.1'].includes(window.location.hostname);
const DEFAULT_CAMERA_MODE = isLocalHost && !import.meta.env.VITE_API_URL ? 'server' : 'browser';

function drawPersons(ctx, persons, scaleX = 1, scaleY = 1) {
  persons.forEach((person) => {
    if (!person.bbox || person.bbox.length < 4) return;
    const [bx1, by1, bx2, by2] = person.bbox;

    const x = bx1 * scaleX;
    const y = by1 * scaleY;
    const w = Math.max(20, (bx2 - bx1) * scaleX);
    const h = Math.max(20, (by2 - by1) * scaleY);

    const isCompliant = person.is_compliant;
    const primaryColor = isCompliant ? '#10b981' : '#ef4444';
    const bgColor = isCompliant ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.18)';

    ctx.fillStyle = bgColor;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    // Corner brackets
    const c = Math.min(12, w / 4);
    ctx.lineWidth = 3;
    [
      [[x, y + c], [x, y], [x + c, y]],
      [[x + w - c, y], [x + w, y], [x + w, y + c]],
      [[x, y + h - c], [x, y + h], [x + c, y + h]],
      [[x + w - c, y + h], [x + w, y + h], [x + w, y + h - c]],
    ].forEach((pts) => {
      ctx.beginPath();
      ctx.moveTo(...pts[0]);
      ctx.lineTo(...pts[1]);
      ctx.lineTo(...pts[2]);
      ctx.stroke();
    });

    const idLabel =
      person.employee_id !== 'UNKNOWN'
        ? `${person.employee_id}: ${person.employee_name}`
        : `UNKNOWN [${person.track_id}]`;
    const statusLabel = isCompliant
      ? 'COMPLIANT'
      : `VIOLATION (No: ${person.missing_items.join(', ') || 'PPE'})`;
    const text = `${idLabel} | ${statusLabel}`;

    ctx.font = 'bold 11px JetBrains Mono, monospace';
    const textWidth = ctx.measureText(text).width;
    const textHeight = 16;
    const badgeY = Math.max(18, y - 6);
    ctx.fillStyle = primaryColor;
    ctx.fillRect(x, badgeY - textHeight, textWidth + 8, textHeight + 4);
    ctx.fillStyle = '#050811';
    ctx.fillText(text, x + 4, badgeY - 3);

    const gearText = `Hardhat: ${person.hardhat ? '✓' : '✗'} | Vest: ${person.safety_vest ? '✓' : '✗'} | Conf: ${(person.detection_confidence * 100).toFixed(0)}%`;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(x, y + h + 2, ctx.measureText(gearText).width + 6, 15);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(gearText, x + 3, y + h + 13);
  });
}

function drawStandbyGrid(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#050811';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
}

export default function LiveMonitoring({
  isActive,
  source,
  persons = [],
  onStart,
  onStop,
  onBrowserStatus,
  isLoading = false,
}) {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const sendingRef = useRef(false);
  const runningRef = useRef(false);
  const browserPersonsRef = useRef([]);
  const flippedRef = useRef(true);

  const [isFlipped, setIsFlipped] = useState(true);
  const [cameraMode, setCameraMode] = useState(DEFAULT_CAMERA_MODE);
  const [browserActive, setBrowserActive] = useState(false);
  const [browserStatusMsg, setBrowserStatusMsg] = useState('');

  const serverActive = isActive && source !== 'browser_camera';
  const streaming = cameraMode === 'browser' ? browserActive : serverActive;

  useEffect(() => {
    flippedRef.current = isFlipped;
  }, [isFlipped]);

  const handleToggleFlip = async () => {
    const next = !isFlipped;
    setIsFlipped(next);
    if (cameraMode === 'server' && serverActive) {
      try {
        await toggleCameraFlip(next, false);
      } catch (err) {
        console.error('Failed to toggle camera flip:', err);
      }
    }
  };

  // -------------------------------------------------------------------------
  // Browser camera mode: capture locally, send frames, draw returned boxes
  // -------------------------------------------------------------------------

  const sendFrame = useCallback(async () => {
    const video = videoRef.current;
    const capture = captureCanvasRef.current;
    if (!runningRef.current || sendingRef.current || !video || !capture || !video.videoWidth) return;

    sendingRef.current = true;
    try {
      const ctx = capture.getContext('2d');
      ctx.save();
      if (flippedRef.current) {
        ctx.translate(capture.width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, 0, 0, capture.width, capture.height);
      ctx.restore();

      const blob = await new Promise((resolve) => capture.toBlob(resolve, 'image/jpeg', 0.7));
      if (!blob || !runningRef.current) return;

      const status = await analyzeFrame(blob);
      if (!runningRef.current) return;
      browserPersonsRef.current = status.persons || [];
      setBrowserStatusMsg('');
      if (onBrowserStatus) onBrowserStatus(status);
    } catch (err) {
      console.warn('Frame analysis failed:', err.message);
      setBrowserStatusMsg('Waiting for server… (a sleeping free-tier server can take ~1 min to wake up)');
      await new Promise((r) => setTimeout(r, 2000));
    } finally {
      sendingRef.current = false;
    }
  }, [onBrowserStatus]);

  const renderLoop = useCallback(function loop() {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas && video.videoWidth) {
      const ctx = canvas.getContext('2d');
      ctx.save();
      if (flippedRef.current) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();
      drawPersons(ctx, browserPersonsRef.current);
    }
    sendFrame();
    rafRef.current = requestAnimationFrame(loop);
  }, [sendFrame]);

  const stopBrowserCamera = useCallback(async () => {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    browserPersonsRef.current = [];
    setBrowserActive(false);
    setBrowserStatusMsg('');
    // Let any in-flight frame finish so it can't re-mark the server as live
    for (let i = 0; i < 30 && sendingRef.current; i += 1) {
      await new Promise((r) => setTimeout(r, 100));
    }
    try {
      await stopMonitoring();
      if (onBrowserStatus) onBrowserStatus(await fetchMonitoringStatus());
    } catch {
      // ignore
    }
  }, [onBrowserStatus]);

  const startBrowserCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      alert('Camera access needs HTTPS and a modern browser.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      const w = CAPTURE_WIDTH;
      const h = Math.round((CAPTURE_WIDTH * vh) / vw);
      // Display canvas and capture canvas share one coordinate space,
      // so boxes from the backend can be drawn without rescaling.
      canvasRef.current.width = w;
      canvasRef.current.height = h;
      captureCanvasRef.current.width = w;
      captureCanvasRef.current.height = h;

      runningRef.current = true;
      setBrowserActive(true);
      setBrowserStatusMsg('Connecting to detection server…');
      rafRef.current = requestAnimationFrame(renderLoop);
    } catch (err) {
      alert(`Could not open camera: ${err.message}`);
    }
  };

  // Stop camera on unmount
  useEffect(() => () => {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
  }, []);

  // Standby / server-mode canvas
  useEffect(() => {
    if (browserActive) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    drawStandbyGrid(ctx, canvas);
    if (persons.length > 0) drawPersons(ctx, persons, 640 / 640, 360 / 480);
  }, [persons, browserActive, serverActive]);

  const handleStart = () => {
    if (cameraMode === 'browser') startBrowserCamera();
    else onStart(0, isFlipped, false);
  };

  const handleStop = () => {
    if (cameraMode === 'browser') stopBrowserCamera();
    else onStop();
  };

  const handleModeChange = () => {
    if (streaming) return;
    setCameraMode((m) => (m === 'browser' ? 'server' : 'browser'));
  };

  const btnStyle = (on) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.35rem',
    backgroundColor: on ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255, 255, 255, 0.05)',
    border: `1px solid ${on ? '#3b82f6' : 'rgba(255, 255, 255, 0.15)'}`,
    color: on ? '#60a5fa' : '#94a3b8',
    cursor: 'pointer',
    borderRadius: '6px',
    padding: '0.35rem 0.65rem',
    fontSize: '0.8rem',
    fontWeight: 500,
  });

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <Camera size={18} />
          <span>Real-Time Sensor Viewport</span>
        </div>
        <div className="panel-controls" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            className="btn btn-sm"
            onClick={handleModeChange}
            disabled={streaming}
            title={
              cameraMode === 'browser'
                ? "Using this device's camera (works when deployed). Click to use the server's camera."
                : "Using the server's camera (local only). Click to use this device's camera."
            }
            style={btnStyle(cameraMode === 'browser')}
          >
            {cameraMode === 'browser' ? <Laptop size={14} /> : <Server size={14} />}
            <span>{cameraMode === 'browser' ? 'This Device Camera' : 'Server Camera'}</span>
          </button>

          <button
            className="btn btn-sm"
            onClick={handleToggleFlip}
            title={isFlipped ? 'Camera is mirrored (click to disable)' : 'Camera is normal (click to mirror)'}
            style={btnStyle(isFlipped)}
          >
            <FlipHorizontal size={14} />
            <span>{isFlipped ? 'Mirrored (Flipped)' : 'Normal'}</span>
          </button>

          {!streaming ? (
            <button className="btn btn-primary btn-sm" onClick={handleStart} disabled={isLoading}>
              <Play size={14} />
              <span>Start Stream</span>
            </button>
          ) : (
            <button className="btn btn-danger btn-sm" onClick={handleStop} disabled={isLoading}>
              <Square size={14} />
              <span>Stop Stream</span>
            </button>
          )}
        </div>
      </div>

      <div className="panel-body" style={{ padding: '0.75rem' }}>
        <div className="viewport-container">
          {/* Hidden elements used for browser-camera capture */}
          <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
          <canvas ref={captureCanvasRef} style={{ display: 'none' }} />

          {cameraMode === 'server' && serverActive ? (
            <img
              src={`${import.meta.env.VITE_API_URL || ''}/monitoring/video_feed`}
              alt="Live Camera Feed"
              className="viewport-stream-img"
              onError={() => console.warn('Video stream reconnecting...')}
            />
          ) : (
            <canvas ref={canvasRef} width={640} height={360} className="viewport-canvas" />
          )}

          {browserActive && browserStatusMsg && (
            <div
              style={{
                position: 'absolute',
                bottom: '0.75rem',
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(15, 23, 42, 0.85)',
                color: '#e2e8f0',
                padding: '0.35rem 0.75rem',
                borderRadius: '6px',
                fontSize: '0.75rem',
              }}
            >
              {browserStatusMsg}
            </div>
          )}

          {!streaming && persons.length === 0 && (
            <div className="viewport-idle-state" style={{ position: 'absolute' }}>
              <Video className="viewport-idle-icon" />
              <div>
                <strong>Sensor Stream Standby</strong>
                <p style={{ marginTop: '0.25rem', fontSize: '0.75rem' }}>
                  {cameraMode === 'browser'
                    ? 'Click "Start Stream" to use this device\'s camera.'
                    : 'Click "Start Stream" to use the camera attached to the server.'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
