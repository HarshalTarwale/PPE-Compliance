import React, { useRef, useEffect } from 'react';
import { Play, Square, Camera, Video, AlertCircle } from 'lucide-react';

export default function LiveMonitoring({
  isActive,
  source,
  persons = [],
  telemetry = {},
  totalTracked = 0,
  compliantCount = 0,
  nonCompliantCount = 0,
  onStart,
  onStop,
  isLoading = false,
}) {
  const canvasRef = useRef(null);

  // Reference video/image frame resolution for coordinates (640x480 standard YOLO coordinate space)
  const BASE_WIDTH = 640;
  const BASE_HEIGHT = 480;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid background representing live sensory feed field-of-view
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Scale factor from base detection coordinates to current canvas resolution
    const scaleX = canvas.width / BASE_WIDTH;
    const scaleY = canvas.height / BASE_HEIGHT;

    // Draw center crosshairs
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 20, cy);
    ctx.lineTo(cx + 20, cy);
    ctx.moveTo(cx, cy - 20);
    ctx.lineTo(cx, cy + 20);
    ctx.stroke();

    if (!isActive && persons.length === 0) {
      return;
    }

    // Render detected worker bounding boxes and identity badges
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

      // Bounding box fill and stroke
      ctx.fillStyle = bgColor;
      ctx.fillRect(x, y, w, h);

      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Corner target brackets
      const cornerLen = Math.min(12, w / 4);
      ctx.lineWidth = 3;
      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLen, y);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerLen);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerLen, y + h);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerLen);
      ctx.stroke();

      // Text Badge
      const idLabel =
        person.employee_id !== 'UNKNOWN'
          ? `${person.employee_id}: ${person.employee_name}`
          : `UNKNOWN [${person.track_id}]`;

      const statusLabel = isCompliant
        ? 'COMPLIANT'
        : `VIOLATION (No: ${person.missing_items.join(', ') || 'PPE'})`;

      const text = `${idLabel} | ${statusLabel}`;
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      const metrics = ctx.measureText(text);
      const textWidth = metrics.width;
      const textHeight = 16;

      const badgeY = Math.max(18, y - 6);
      ctx.fillStyle = primaryColor;
      ctx.fillRect(x, badgeY - textHeight, textWidth + 8, textHeight + 4);

      ctx.fillStyle = '#050811';
      ctx.fillText(text, x + 4, badgeY - 3);

      // Equipment indicators under box
      const gearText = `Hardhat: ${person.hardhat ? '✓' : '✗'} | Vest: ${person.safety_vest ? '✓' : '✗'} | Conf: ${(person.detection_confidence * 100).toFixed(0)}%`;
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      const gearMetrics = ctx.measureText(gearText);
      ctx.fillRect(x, y + h + 2, gearMetrics.width + 6, 15);
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(gearText, x + 3, y + h + 13);
    });
  }, [persons, isActive]);

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <Camera size={18} />
          <span>Real-Time Sensor Viewport</span>
        </div>
        <div className="panel-controls">
          {!isActive ? (
            <button
              className="btn btn-primary btn-sm"
              onClick={() => onStart(0)}
              disabled={isLoading}
            >
              <Play size={14} />
              <span>Start Stream</span>
            </button>
          ) : (
            <button
              className="btn btn-danger btn-sm"
              onClick={onStop}
              disabled={isLoading}
            >
              <Square size={14} />
              <span>Stop Stream</span>
            </button>
          )}
        </div>
      </div>

      <div className="panel-body" style={{ padding: '0.75rem' }}>
        <div className="viewport-container">
          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            className="viewport-canvas"
          />

          {!isActive && persons.length === 0 && (
            <div className="viewport-idle-state" style={{ position: 'absolute' }}>
              <Video className="viewport-idle-icon" />
              <div>
                <strong>Sensor Stream Standby</strong>
                <p style={{ marginTop: '0.25rem', fontSize: '0.75rem' }}>
                  Click "Start Stream" above or connect an RTSP / webcam source.
                </p>
              </div>
            </div>
          )}

          {/* Real-Time Telemetry HUD Overlay */}
          <div className="viewport-hud">
            <div className="hud-row">
              <span className="hud-label">STATUS</span>
              <span
                className="hud-value"
                style={{ color: isActive ? 'var(--color-compliant)' : 'var(--text-muted)' }}
              >
                {isActive ? 'RECORDING' : 'IDLE'}
              </span>
            </div>
            <div className="hud-row">
              <span className="hud-label">FPS / LATENCY</span>
              <span className="hud-value">
                {telemetry.fps > 0 ? telemetry.fps.toFixed(1) : '0.0'} FPS (
                {telemetry.total_ms > 0 ? telemetry.total_ms.toFixed(1) : '0.0'} ms)
              </span>
            </div>
            <div className="hud-row">
              <span className="hud-label">PERSONNEL</span>
              <span className="hud-value">
                {totalTracked} (Comp: {compliantCount} | Viol: {nonCompliantCount})
              </span>
            </div>
            {source && (
              <div className="hud-row">
                <span className="hud-label">SOURCE</span>
                <span className="hud-value">{source}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
