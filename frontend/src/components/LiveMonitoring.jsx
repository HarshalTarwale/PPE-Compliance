import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Play, Square, Camera, Video, FlipHorizontal, Laptop, Server } from 'lucide-react';
import { toggleCameraFlip, analyzeFrame, stopMonitoring, fetchMonitoringStatus } from '../api/client';

// Width of frames sent to the backend (height follows the camera aspect ratio).
// Kept small so each upload is quick on slow connections to a cloud server.
const CAPTURE_WIDTH = 480;
const CAPTURE_JPEG_QUALITY = 0.6;
// Width of the on-screen canvas; backend boxes are scaled up to it
const DISPLAY_WIDTH = 640;
// The first frame may have to wait for a sleeping free-tier server to boot
const FIRST_FRAME_TIMEOUT_MS = 90000;
const FRAME_TIMEOUT_MS = 30000;
// No answer for this long on the first frame means the server is waking up
const WAKING_HINT_AFTER_MS = 5000;

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

// Average brightness (0-255) of a canvas, sampled sparsely so it stays cheap
function averageBrightness(ctx, width, height) {
  const { data } = ctx.getImageData(0, 0, width, height);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 4 * 16) {
    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    count += 1;
  }
  return count ? sum / count : 255;
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
  const startedAtRef = useRef(0);
  const gotFirstFrameRef = useRef(false);

  const [isFlipped, setIsFlipped] = useState(true);
  const [cameraMode, setCameraMode] = useState(DEFAULT_CAMERA_MODE);
  const [browserActive, setBrowserActive] = useState(false);
  const [browserStatusMsg, setBrowserStatusMsg] = useState('');
  const [isTooDark, setIsTooDark] = useState(false);
  const [frameMs, setFrameMs] = useState(null);

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

      // A nearly black picture usually means the lens is covered, a privacy
      // shutter is closed, or another app is holding the camera.
      setIsTooDark(averageBrightness(ctx, capture.width, capture.height) < 30);

      const blob = await new Promise((resolve) =>
        capture.toBlob(resolve, 'image/jpeg', CAPTURE_JPEG_QUALITY),
      );
      if (!blob || !runningRef.current) return;

      // Generous timeout for the first frame: a sleeping free-tier server needs ~1 min to wake up
      const controller = new AbortController();
      const timer = setTimeout(
        () => controller.abort(),
        gotFirstFrameRef.current ? FRAME_TIMEOUT_MS : FIRST_FRAME_TIMEOUT_MS,
      );
      const sentAt = performance.now();
      let status;
      try {
        status = await analyzeFrame(blob, controller.signal);
      } finally {
        clearTimeout(timer);
      }
      if (!runningRef.current) return;
      gotFirstFrameRef.current = true;
      setFrameMs(Math.round(performance.now() - sentAt));
      browserPersonsRef.current = status.persons || [];
      setBrowserStatusMsg('');
      if (onBrowserStatus) onBrowserStatus(status);
    } catch (err) {
      console.warn('Frame analysis failed:', err.message);
      // Old boxes would point at where people were before the outage
      browserPersonsRef.current = [];
      setBrowserStatusMsg(
        gotFirstFrameRef.current
          ? 'Lost connection to the detection server. Retrying…'
          : 'Detection server is not answering yet (it may be starting up). Retrying…',
      );
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
      const scale = canvas.width / captureCanvasRef.current.width;
      drawPersons(ctx, browserPersonsRef.current, scale, scale);
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
    setIsTooDark(false);
    setFrameMs(null);
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
      // Same aspect ratio on both canvases, so backend boxes only need one scale factor
      canvasRef.current.width = DISPLAY_WIDTH;
      canvasRef.current.height = Math.round((DISPLAY_WIDTH * vh) / vw);
      captureCanvasRef.current.width = CAPTURE_WIDTH;
      captureCanvasRef.current.height = Math.round((CAPTURE_WIDTH * vh) / vw);

      runningRef.current = true;
      gotFirstFrameRef.current = false;
      startedAtRef.current = Date.now();
      setFrameMs(null);
      setBrowserActive(true);
      setBrowserStatusMsg('Connecting to detection server…');
      rafRef.current = requestAnimationFrame(renderLoop);
    } catch (err) {
      alert(`Could not open camera: ${err.message}`);
    }
  };

  // While waiting for the first answer, explain a slow start instead of a bare "Connecting…"
  useEffect(() => {
    if (!browserActive) return undefined;
    const timer = setInterval(() => {
      if (gotFirstFrameRef.current) return;
      const waited = Date.now() - startedAtRef.current;
      if (waited > WAKING_HINT_AFTER_MS) {
        setBrowserStatusMsg(
          `Waking up the detection server (${Math.round(waited / 1000)} s). Free cloud hosting sleeps when idle; this can take up to a minute…`,
        );
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [browserActive]);

  // Stop camera on unmount (e.g. switching tabs) and tell the server the stream ended
  useEffect(() => () => {
    const wasRunning = runningRef.current;
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (wasRunning) stopMonitoring().catch(() => {});
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
    if (streaming || !isLocalHost) return;
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
          {isLocalHost && (
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
          )}

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

          {browserActive && isTooDark && (
            <div
              style={{
                position: 'absolute',
                top: '0.75rem',
                left: '50%',
                transform: 'translateX(-50%)',
                maxWidth: '90%',
                background: 'rgba(234, 179, 8, 0.92)',
                color: '#111827',
                padding: '0.5rem 0.85rem',
                borderRadius: '6px',
                fontSize: '0.8rem',
                fontWeight: 600,
                textAlign: 'center',
              }}
            >
              Camera image is almost black. Open the laptop&apos;s camera privacy shutter / uncover the
              lens, turn on room lights, and close other apps or tabs using the camera
              (Zoom, Teams, the Windows Camera app, a local server).
            </div>
          )}

          {browserActive && frameMs !== null && !browserStatusMsg && (
            <div
              className="mono"
              title="Round-trip time for one frame to the detection server and back"
              style={{
                position: 'absolute',
                bottom: '0.75rem',
                right: '0.75rem',
                background: 'rgba(15, 23, 42, 0.85)',
                color: frameMs > 3000 ? '#fbbf24' : '#94a3b8',
                padding: '0.2rem 0.5rem',
                borderRadius: '4px',
                fontSize: '0.7rem',
              }}
            >
              Server: {(frameMs / 1000).toFixed(1)} s/frame
            </div>
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
