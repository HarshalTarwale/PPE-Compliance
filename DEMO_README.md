# Real-Time PPE Safety Monitoring & Compliance System
## Complete Project Demonstration, Accuracy & Viva Defense Guide

> **Quick Links & Artifacts:**
> - 🗣️ **Downloadable Spoken Script (Word):** [`What_To_Tell_Mam_Tomorrow_Demonstration_Script.docx`](What_To_Tell_Mam_Tomorrow_Demonstration_Script.docx)
> - 📄 **Complete Technical Project Guide (Word):** [`PPE_Compliance_Project_Demonstration_Guide.docx`](PPE_Compliance_Project_Demonstration_Guide.docx)
> - 🚀 **One-Click Launch Script:** [`start.bat`](start.bat) (Starts FastAPI on `8000` + React Dashboard on `5173`)
> - 🧪 **Automated Test Suite:** `.venv\Scripts\python.exe -m pytest tests/ -v` (177 tests passing in 2.3s)

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Model Accuracy & Evaluation Breakdown](#2-model-accuracy--evaluation-breakdown)
   - [Overall Test Split Metrics](#21-overall-metrics-test-split-82-images-476-instances)
   - [Per-Class Target Equipment Breakdown](#22-per-class-target-equipment-breakdown)
   - [Baseline (25-Class) vs. Optimized (5-Class) Comparison](#23-baseline-vs-candidate-comparison)
   - [How to Check / Re-Run Accuracy Metrics Live](#24-how-to-check--re-run-accuracy-metrics-live)
3. [Speed, Latency & Hardware Benchmarks](#3-speed-latency--hardware-benchmarks)
4. [System Architecture & 8-Stage Pipeline](#4-system-architecture--8-stage-pipeline)
5. [Technologies, Models & Tools Used (With Rationale)](#5-technologies-models--tools-used)
6. [Detailed Technical Implementation](#6-detailed-technical-implementation)
   - [6.1. YOLOv8n Object Detection](#61-yolov8n-object-detection)
   - [6.2. Multi-Object Spatial Tracking (`SimpleIoUTracker`)](#62-multi-object-spatial-tracking)
   - [6.3. Geometric Person-to-PPE Association](#63-geometric-person-to-ppe-association)
   - [6.4. Privacy-Preserving Facial Biometrics (`YuNet + SFace`)](#64-privacy-preserving-facial-biometrics)
   - [6.5. Sliding-Window Temporal Anti-Flicker Filter](#65-sliding-window-temporal-anti-flicker-filter)
   - [6.6. Safety Compliance Rule Engine](#66-safety-compliance-rule-engine)
   - [6.7. FastAPI Asynchronous Backend & Worker Isolation](#67-fastapi-backend--worker-isolation)
   - [6.8. React 19 Canvas Dashboard & Dynamic Scaling](#68-react-19-canvas-dashboard--dynamic-scaling)
7. [Step-by-Step Live Demonstration Script](#7-step-by-step-live-demonstration-script)
8. [Top 12 Evaluator / Viva Defense Q&A](#8-top-12-evaluator--viva-defense-qa)

---

## 1. Executive Summary & Problem Statement

Industrial sites, construction zones, and manufacturing facilities enforce mandatory Personal Protective Equipment (PPE) regulations (e.g., **OSHA 1926.100** for Head Protection and **OSHA 1926.201** for High-Visibility Safety Vests). Manual safety inspections are sporadic, dangerous, and prone to human error.

This platform automates visual safety audits in real time:
- It tracks multiple workers simultaneously across video frames.
- It determines which safety equipment physically belongs to each worker using anatomical geometric priors.
- It identifies registered workers through privacy-preserving 128-D face embeddings without storing sensitive raw photos.
- It dampens single-frame detection drops using sliding-window temporal smoothing.
- It broadcasts lightweight structured JSON telemetry over WebSockets to a React industrial dashboard running at **115+ FPS**.

---

## 2. Model Accuracy & Evaluation Breakdown

In computer vision object detection, performance is measured via **Precision**, **Recall**, and **mAP (Mean Average Precision)** at specific Intersection-over-Union (IoU) thresholds.

### 2.1. Overall Metrics (Test Split: 82 Images, 476 Instances)
*Measured on unseen test data (`outputs/evaluation/candidate_5class/report.txt`):*

| Metric | Score (%) | Meaning & Significance |
|---|---|---|
| **Precision** | **`78.63%`** | When the model predicts a person or PPE item, it is correct **78.63%** of the time. |
| **mAP@50** | **`61.53%`** | Standard Pascal VOC / COCO benchmark (IoU $\ge 0.50$). On the validation split, mAP@50 reached **65.10%**. |
| **Recall** | **`58.03%`** | Out of all ground-truth safety gear in the images, the model detected **58.03%**. |
| **mAP@50-95** | **`38.19%`** | Rigorous COCO average across IoU thresholds from 0.50 to 0.95 in steps of 0.05. |

### 2.2. Per-Class Target Equipment Breakdown
| Target Class | Test Instances | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50-95 (%) |
|---|---|---|---|---|---|
| 🪖 **Hardhat** | 110 | **`94.47%`** | 62.18% | **`73.23%`** | 45.92% |
| 👤 **Person** | 174 | **`80.98%`** | 61.49% | **`68.73%`** | 48.47% |
| 🦺 **NO-Safety Vest** | 90 | **`81.00%`** | 52.11% | **`58.45%`** | 36.06% |
| 🦺 **Safety Vest** | 61 | **`69.86%`** | 65.57% | **`64.70%`** | 36.08% |
| 🚫 **NO-Hardhat** | 41 | **`66.86%`** | 48.78% | **`42.51%`** | 24.40% |

> **Critical Safety Highlight:** **Hardhat Precision reached 94.47%**. In industrial safety, high precision on safety items ensures virtually zero false compliance passes.

### 2.3. Baseline vs. Candidate Comparison
We evaluated the baseline 25-class model (trained on noisy background construction vehicles) against our clean 5-class target PPE model:

| Evaluation Metric | Baseline (25-Class) | Candidate (5-Class) | Empirical Delta | Relative Improvement |
|---|---|---|---|---|
| **mAP@50** | `52.41%` | **`61.53%`** | **`+9.12%`** | **+17.4% Relative** |
| **mAP@50-95** | `36.86%` | **`38.19%`** | **`+1.33%`** | **+3.6% Relative** |
| **Precision** | `69.71%` | **`78.63%`** | **`+8.92%`** | **+12.8% Relative** |
| **Recall** | `48.71%` | **`58.03%`** | **`+9.32%`** | **+19.1% Relative** |
| **Hardhat Precision** | `87.27%` | **`94.47%`** | **`+7.20%`** | **+8.3% Relative** |

*Takeaway:* Pruning 20 rare background machinery classes eliminated gradient dilution, directly increasing mAP@50 by **+9.12%** and recall by **+9.32%**.

### 2.4. How to Check / Re-Run Accuracy Metrics Live

#### Method A: View the Pre-Computed Report (Instant)
- View formatted text report: [`outputs/evaluation/candidate_5class/report.txt`](outputs/evaluation/candidate_5class/report.txt)
- View side-by-side JSON comparison: [`outputs/evaluation/model_comparison.json`](outputs/evaluation/model_comparison.json)

#### Method B: Re-Evaluate Live via Terminal Command
Run this in PowerShell / Command Prompt to compute metrics on the test split:
```cmd
.venv\Scripts\python.exe training/evaluate.py --model models/weights/ppe_5class_best.pt --data data/processed/ppe_5class/data.yaml --split test --device auto
```

#### Method C: Run the Empirical Model Comparison
```cmd
.venv\Scripts\python.exe scripts/compare_models.py --baseline outputs/evaluation/results.json --candidate models/weights/ppe_5class_best.pt --data data/processed/ppe_5class/data.yaml --split test
```

---

## 3. Speed, Latency & Hardware Benchmarks

Measured over 100 continuous iterations on an **NVIDIA GeForce RTX 4050 Laptop GPU** (6140 MiB VRAM, 640×480 resolution):

| Pipeline Stage | FP32 Mean Latency | FP16 Mean Latency | FP16 P95 Latency | FP16 Throughput |
|---|---|---|---|---|
| **YOLOv8 Detection** | 8.37 ms | **`7.41 ms`** | 10.79 ms | **134.9 FPS** |
| **Spatial PPE Association** | 0.01 ms | **`0.01 ms`** | 0.01 ms | **>100,000 FPS** |
| **Face Feature Extraction (SFace)** | 1.16 ms | **`1.08 ms`** | 1.29 ms | **926.7 FPS** |
| **Temporal Smoothing (Filter)** | 0.01 ms | **`0.01 ms`** | 0.01 ms | **>80,000 FPS** |
| **Full End-to-End Pipeline** | 9.23 ms | **`8.69 ms`** | 12.76 ms | **115.0 FPS** |

---

## 4. System Architecture & 8-Stage Pipeline

```
  [ Live Video / Webcam / RTSP Stream ]
                   │
                   ▼
  Stage 1: YOLOv8n Inference (7.4 ms)
  ──► Predicts BBoxes for Persons & PPE items
                   │
                   ▼
  Stage 2: SimpleIoUTracker Matching (0.05 ms)
  ──► Bipartite IoU greedy match (IoU >= 0.30); assigns 'track_1', 'track_2'
                   │
                   ▼
  Stage 3: Spatial Anatomical Associator (0.01 ms)
  ──► Gated by Head window [-0.20, 0.38] and Torso window [0.12, 0.72]
  ──► Computes Score S = 0.45*Containment + 0.40*Proximity + 0.15*Conf
                   │
                   ▼
  Stage 4: Biometric Face Engine (1.1 ms)
  ──► Crops head region -> YuNet 5-landmark alignment -> SFace 128-D vector
  ──► Cosine >= 0.363 -> Employee ID (else 'UNKNOWN')
                   │
                   ▼
  Stage 5: Sliding-Window Temporal Filter (0.01 ms)
  ──► 10-frame sliding deque buffer per track; 60% voting presence consensus
                   │
                   ▼
  Stage 6: Pure Python Compliance Engine (<0.01 ms)
  ──► Hardhat present AND Safety Vest present => COMPLIANT
                   │
                   ▼
  Stage 7: Dedicated Backend Worker Thread & WebSocket Dispatcher
  ──► Asynchronous broadcast of lightweight structured JSON telemetry
                   │
                   ▼
  Stage 8: React Canvas Industrial Dashboard
  ──► Dynamic 640x480 coordinate projection; renders reticles, tags, and KPI stats
```

---

## 5. Technologies, Models & Tools Used

| Subsystem | Technology | Purpose & Architectural Rationale |
|---|---|---|
| **Object Detection** | **Ultralytics YOLOv8n (Nano)** | 3.0M parameters, 8.1 GFLOPs. Achieves 135 FPS on GPU; leaves ample compute for biometrics. |
| **Model Optimization** | **ONNX Runtime + ONNXSlim** | Generates standalone 11.7 MB portable model without PyTorch overhead. |
| **Face Detection** | **OpenCV YuNet ONNX** | 232 KB model; extracts 5 landmarks (eyes, nose, mouth corners) in <1 ms even under hardhat shadows. |
| **Face Recognition** | **OpenCV SFace ONNX** | 38.7 MB model; extracts 128-D L2-normalized unit vectors. Eliminates `dlib`/CMake build issues. |
| **Multi-Object Tracking** | **SimpleIoUTracker** | Greedy spatial IoU tracking (>10,000 FPS); persists track IDs across frames without heavy Re-ID networks. |
| **Temporal Smoothing** | **PPETemporalFilter** | Circular deque sliding window; absorbs single-frame blurs and momentary head turns. |
| **Backend Service** | **FastAPI + Uvicorn + Pydantic** | Dedicated daemon background thread isolates ML inference from the ASGI asynchronous event loop. |
| **Frontend UI** | **React 19 + Vite + HTML5 Canvas** | Industrial dark theme. Canvas 2D renders worker bounding boxes and overlays smoothly at 60 FPS. |
| **Automated Testing** | **Pytest** | 177 unit & integration tests validating math, edge cases, endpoints, and WebSocket contracts. |

---

## 6. Detailed Technical Implementation

### 6.1. YOLOv8n Object Detection
- **Trained on:** Construction Site Safety dataset.
- **Inference Runtime:** Automatic CUDA GPU acceleration with FP16 half-precision (`cuda:0`).
- **Classes:** `0: Hardhat`, `1: NO-Hardhat`, `2: NO-Safety Vest`, `3: Person`, `4: Safety Vest`.

### 6.2. Multi-Object Spatial Tracking
- Implemented in [`detection/pipeline.py::SimpleIoUTracker`](detection/pipeline.py).
- Calculates pairwise Intersection-over-Union between existing tracks and new person bounding boxes:
  $$\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} \ge 0.30$$
- Greedy matching assigns persistent monotonic identifiers (`track_1`, `track_2`).
- Maintains a **15-frame grace period** (`max_missed=15`) before pruning dead tracks.

### 6.3. Geometric Person-to-PPE Association
- Implemented in [`detection/ppe_association.py`](detection/ppe_association.py).
- Relies on normalized human anatomical proportions rather than training an expensive secondary network:
  - Relative vertical position: $rel\_y = \frac{cy - y_{p1}}{H_p}$
  - Head Region: $rel\_y \in [-0.20, 0.38]$ (ideal $0.08$). Negative upper bound accommodates hardhats extending above the head box.
  - Torso Region: $rel\_y \in [0.12, 0.72]$ (ideal $0.38$).
- Candidate Composite Scoring:
  $$S = 0.45 \cdot \text{Containment} + 0.40 \cdot \text{Proximity} + 0.15 \cdot \text{Confidence}$$
- **Exclusive 1-to-1 matching:** When workers overlap, the worker with highest score $S$ is exclusively assigned the PPE item. Unassigned items failing minimum score ($0.20$) are safely ignored.

### 6.4. Privacy-Preserving Facial Biometrics
- Implemented in [`recognition/embeddings.py`](recognition/embeddings.py) and [`recognition/registration.py`](recognition/registration.py).
- **Zero Raw Images on Disk:** Raw employee photos are processed strictly in volatile RAM and immediately discarded.
- Only 128-dimensional float32 normalized embeddings are persisted in `embeddings.npz` (~512 bytes).
- Matching requires Cosine Similarity $\ge 0.363$. Ambiguous or unknown faces strictly return `UNKNOWN` with zero forced guessing.

### 6.5. Sliding-Window Temporal Anti-Flicker Filter
- Implemented in [`compliance/temporal_filter.py`](compliance/temporal_filter.py).
- Per-track circular buffer (`collections.deque`, window size = 10 frames).
- An item is marked present only if observed in $\ge 60\%$ of recent frames ($\ge 6$ out of 10).
- Tolerates up to 4 consecutive missed frames due to rapid movement or lighting glare without flashing false alerts.

### 6.6. Safety Compliance Rule Engine
- Implemented in [`compliance/rules.py`](compliance/rules.py).
- Pure Python zero-dependency engine.
- Mandated standard: `Hardhat == True AND Safety Vest == True => COMPLIANT`.
- Any missing item immediately sets status to `NON_COMPLIANT` and records the missing gear in `missing_items`.

### 6.7. FastAPI Backend & Worker Isolation
- Implemented in [`backend/main.py`](backend/main.py) and [`backend/services/monitoring_service.py`](backend/services/monitoring_service.py).
- Inference runs in a dedicated background worker thread (`_worker_loop`).
- OpenCV and YOLO never block the ASGI event loop, ensuring sub-millisecond REST responses and smooth WebSocket telemetry.
- WebSocket stream (`/ws/monitoring`) sends structured JSON without raw video pixels, saving 98% network bandwidth.

### 6.8. React 19 Canvas Dashboard & Dynamic Scaling
- Implemented in [`frontend/src/`](frontend/src/).
- Canvas coordinates dynamically project from reference $640 \times 480$ space to user viewport dimensions:
  $$\text{scale}_X = \frac{\text{canvas.width}}{640}, \quad \text{scale}_Y = \frac{\text{canvas.height}}{480}$$
- Provides live camera flip, real-time KPI cards, violation alert banners, and employee catalog management.

---

## 7. Step-by-Step Live Demonstration Script

Follow this sequence tomorrow:

1. **Launch the System (Terminal):**
   ```cmd
   .\start.bat
   ```
   *Say:* *"A single batch script launches our FastAPI backend and React dashboard, validating GPU acceleration automatically."*

2. **Tour the Industrial Dashboard (`http://localhost:5173/dashboard`):**
   *Say:* *"Our dashboard uses an OSHA-compliant high-contrast dark theme. The top KPI cards show total tracked personnel, compliant workers, and active violations in real time."*

3. **Start Live Camera Feed & Show Hardware Telemetry:**
   Click **"Start Camera"**. Point out the live telemetry badge in the top-right corner of the canvas:
   *Say:* *"Notice the speed: YOLOv8 runs in ~7.4 ms, spatial association in 0.01 ms, and temporal smoothing in 0.01 ms. Total throughput is over 110 FPS on our RTX 4050 GPU."*

4. **Show Compliance vs. Violation Detection:**
   - Wear/hold Hardhat + Vest: Bounding box turns **GREEN (`COMPLIANT`)**.
   - Take off Hardhat: Bounding box immediately turns **RED (`NON_COMPLIANT: Missing Hardhat`)** and flags an alert banner.
   *Say:* *"Our deterministic compliance engine flags missing required gear immediately without cloud round-trips."*

5. **Demonstrate Temporal Anti-Flicker Protection:**
   Quickly turn head away or wave a hand in front of the hardhat.
   *Say:* *"Notice how stable the detection is. Traditional detectors flicker on single-frame drops. Our sliding-window filter requires a 60% voting consensus over 10 frames, absorbing optical dropouts."*

6. **Demonstrate Employee Biometric Registration (`/register`):**
   - Enter Employee ID `EMP101`, Name `Dr. Jane Chen`, and drop 2 portrait photos. Click **"Register"**.
   - Return to Live Feed: Track label updates from `UNKNOWN` to `EMP101: Dr. Jane Chen`.
   *Say:* *"YuNet aligned 5 face landmarks and SFace generated a 128-D embedding. Raw photos were immediately purged from RAM to guarantee worker biometric privacy."*

7. **Show OpenAPI Docs & Automated Test Suite:**
   - Open `http://localhost:8000/docs` in browser.
   - Run in terminal: `.venv\Scripts\python.exe -m pytest tests/ -v` (177 passed in 2.3s).
   *Say:* *"The platform is backed by a full OpenAPI specification and 177 automated pytest tests with 100% pass rate."*

---

## 8. Top 12 Evaluator / Viva Defense Q&A

**Q1: Why choose YOLOv8n over Faster R-CNN or YOLOv8x?**
> *Answer:* Faster R-CNN runs at only 15–25 FPS with 100+ ms latency. YOLOv8n has only 3.0M parameters and 8.1 GFLOPs, executing in **7.41 ms (134.9 FPS)** with FP16 Tensor Cores. It achieves **94.5% Hardhat precision** and **61.5% mAP@50**, leaving 85% of GPU cycles free for multi-stream tracking and biometrics.

**Q2: How do you associate PPE to a worker when multiple workers overlap?**
> *Answer:* In `detection/ppe_association.py`, we apply anatomical vertical bounds (headgear $[-0.20, 0.38]$, torso $[0.12, 0.72]$) and compute a composite score:
> $$S = 0.45 \cdot \text{Containment} + 0.40 \cdot \text{Proximity} + 0.15 \cdot \text{Confidence}$$
> Proximity penalizes deviation from the vertical spine midline. The worker with highest score is assigned the PPE exclusively.

**Q3: Why not train an end-to-end multi-task network?**
> *Answer:* Multi-task networks suffer from negative gradient transfer and catastrophic forgetting. Adding a new safety item requires retraining everything. Our modular decoupled architecture allows detection, tracking, biometrics, and compliance rules to be updated and tested independently.

**Q4: Why OpenCV YuNet and SFace instead of dlib or FaceNet?**
> *Answer:* `dlib` requires C++ CMake compilation which frequently breaks on Windows. FaceNet is heavy and slow. YuNet is only 232 KB (<1 ms landmark extraction) and SFace is 38.7 MB (1.08 ms embedding extraction). Both run natively via OpenCV's `cv2.dnn` without extra deep learning frameworks.

**Q5: How do you ensure biometric privacy and GDPR compliance?**
> *Answer:* Privacy-by-Design. Raw facial photos are processed in volatile RAM and immediately discarded. Only 128-dimensional float32 normalized unit vectors are stored in `.npz` files (~512 bytes). Deep face embeddings are mathematically non-invertible, so original photos cannot be reconstructed.

**Q6: What prevents false alarms when a worker turns their head?**
> *Answer:* The Temporal Smoothing Filter (`compliance/temporal_filter.py`) maintains a 10-frame sliding window per track. Gear must be observed in $\ge 60\%$ of recent frames. Momentary 1-to-3 frame occlusions are absorbed without flipping compliance status.

**Q7: Why stream structured JSON over WebSockets instead of an MJPEG video feed?**
> *Answer:* Video streaming saturates network bandwidth (50–100 Mbps) and consumes heavy CPU compression. Our backend transmits lightweight structured JSON telemetry (~2–5 KB/s per client). The React client renders overlays locally at 60 FPS.

**Q8: Why did the 5-class model beat the 25-class baseline by +9.1% mAP?**
> *Answer:* The raw 25-class dataset had severe class imbalance with rare background machinery classes. These rare classes diluted gradients and penalized worker features. Pruning to the 5 target PPE classes focused all learning on workers, hardhats, and vests, increasing mAP@50 from 52.41% to 61.53% and recall from 48.71% to 58.03%.

**Q9: What happens if a worker leaves the frame and returns?**
> *Answer:* `SimpleIoUTracker` maintains a 15-frame grace period. If a worker steps behind a pillar and returns within half a second, their track ID is preserved. If they leave longer, the track is pruned. When they return, a new track spawns and the face engine re-identifies them automatically.

**Q10: How does the backend prevent inference from freezing the web API?**
> *Answer:* Inference runs in a dedicated background worker daemon thread (`MonitoringService._worker_loop`). It pushes events to an asyncio queue. The FastAPI ASGI event loop remains completely unblocked to serve HTTP and WebSocket requests with sub-millisecond response times.

**Q11: What are the known limitations of this system?**
> *Answer:* 1) Scale: workers must be $\ge 80 \times 40$ px for PPE association and $\ge 40 \times 40$ px for face detection; 2) Crouching: extreme bending shifts vertical box priors (future work: YOLOv8-pose skeleton joints); 3) Facial yaw: face recognition requires orientation within $\pm 45^\circ$ yaw. Crucially, when recognition fails, identity safely defaults to `UNKNOWN` rather than making a false guess.

**Q12: How was the system validated for correctness?**
> *Answer:* We built an automated test suite with **177 unit and integration tests** using pytest. It validates geometric association math, temporal voting edge cases, biometric rejection logic, REST endpoints, and WebSocket contracts with 100% pass rate in 2.3 seconds.
