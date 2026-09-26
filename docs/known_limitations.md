# Known Limitations & Operating Boundaries

This document provides a transparent assessment of the operational constraints, edge cases, and known limitations of the PPE compliance monitoring system.

---

## 1. Computer Vision & Perception Boundaries

### 1.1 Distance & Subject Scale
- **Minimum Resolvable Size**: The system requires a person bounding box of at least **80×40 pixels** for reliable PPE association and at least **40×40 pixels** for face landmark extraction via YuNet.
- **Distant Workers**: When workers appear far away in wide-angle CCTV shots ($>25\text{ meters}$ with standard lenses), hardhats and safety vests occupy very few pixels, leading to lower recall.

### 1.2 Extreme Occlusion & Scaffolding
- **Partial Body Visibility**: The spatial associator uses relative vertical priors ($Y_{rel}$) based on the full person height. If a worker is half-hidden behind a barrier, desk, or scaffolding, the estimated head and torso regions will shift proportionally, potentially degrading association confidence.
- **Crowded Overlaps**: If multiple workers stand in direct line-of-sight overlap, greedy IoU tracking may swap track IDs during the occlusion interval.

### 1.3 Lighting & Environmental Artifacts
- **High-Contrast Glare**: Direct sunlight or industrial spotlight glare reflecting off white hardhats or reflective vest stripes can cause momentary detection drops. The temporal filter (10-frame window) dampens this, but sustained glare will register as missing PPE.
- **Low-Light / Night Operation**: Standard RGB cameras without infrared or auxiliary illumination degrade both YOLO detection and SFace feature extraction.

---

## 2. Face Recognition & Biometric Identification

### 2.1 Head Poses & Obstructions
- **Extreme Yaw / Pitch**: YuNet 5-landmark face alignment requires both eyes and nose to be partially visible. Angles beyond $\pm 45^\circ$ yaw or $\pm 30^\circ$ pitch will yield no face detection.
- **PPE Self-Occlusion**: Full-face shields, dark welding masks, or low-pulled hardhat visors will intentionally prevent face recognition.
- **Safety Policy**: In these cases, the person is **never assigned a false identity**; they strictly remain designated as `UNKNOWN`.

### 2.2 Re-Identification Without Appearance DeepSORT
- The current tracking implementation is a spatial `SimpleIoUTracker` that relies on spatial proximity and bounding box overlap.
- If a tracked worker exits the camera view and returns after `max_missed=15` frames, a new `track_id` is spawned. Their identity is re-acquired only once a clear face angle is detected.

---

## 3. Operational & Network Constraints

### 3.1 Network Bandwidth
- **WebSocket Streaming**: Optimized to send only lightweight structured JSON telemetry (~2-5 KB/s per client).
- **Video Feeds**: For multi-camera facilities, raw RTSP video streams should be decoded at the edge rather than routed through the web server.

### 3.2 Single-Worker Dedicated Threading
- The backend `MonitoringService` runs inference on a single background worker thread. While adequate for single streams up to 115 FPS on modern GPUs, scaling to 8+ simultaneous camera streams requires multi-process worker pools or an asynchronous inference queue (e.g. Triton Inference Server / Celery).
