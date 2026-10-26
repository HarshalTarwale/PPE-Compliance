"""
Script to generate the comprehensive Demonstration & Viva Defense Guide DOCX document
for the Real-Time PPE Safety Monitoring & Compliance System.
"""

import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a table cell in dxa (1 pt = 20 dxa)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def add_callout(doc, text, title="KEY TAKEAWAY FOR DEMO", border_color="1E3A8A", bg_color="F1F5F9"):
    """Adds a callout quote box with left colored border and tinted background."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=200, right=200)
    
    # Left border only
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(tc_borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    
    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table_headers(table, col_widths, headers, bg_color="1E3A8A"):
    """Applies styles to table header row."""
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], bg_color)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

def style_data_row(row, col_widths, values, is_even=False):
    """Styles a single data row with optional alternating row color."""
    bg_color = "F8FAFC" if is_even else "FFFFFF"
    for i, val in enumerate(values):
        cell = row.cells[i]
        cell.text = str(val)
        cell.width = col_widths[i]
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=80, bottom=80, left=150, right=150)
        p = cell.paragraphs[0]
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

def build_demo_guide():
    doc = Document()
    
    # Page Margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)
    
    # -------------------------------------------------------------
    # TITLE & METADATA
    # -------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    run_title = title_p.add_run("Real-Time PPE Safety Monitoring & Compliance System")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A) # Slate 900
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    run_sub = sub_p.add_run("Project Demonstration, Technical Architecture & Viva Defense Master Guide")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(14)
    run_sub.font.color.rgb = RGBColor(0x25, 0x63, 0xEB) # Blue 600
    run_sub.font.bold = True

    # Metadata table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_widths = [Inches(2.2), Inches(4.3)]
    meta_data = [
        ("Project Classification", "Deep Learning & Real-Time Computer Vision Capstone System"),
        ("Core Capabilities", "YOLOv8n Detection, IoU Tracking, Spatial Prior Association, YuNet/SFace Face Biometrics, Temporal Anti-Flicker Filter, FastAPI & React Dashboard"),
        ("Target Evaluation Date", "Demonstration Ready (Full 177 Unit & Integration Tests Passing)"),
        ("Live Demonstration Mode", "One-Click Instant Execution via start.bat (Webcam / Video / Static Test)"),
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[r_idx]
        row.cells[0].text = k
        row.cells[0].paragraphs[0].runs[0].font.bold = True
        row.cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        row.cells[0].width = meta_widths[0]
        row.cells[1].text = v
        row.cells[1].width = meta_widths[1]
        set_cell_background(row.cells[0], "F1F5F9")
        set_cell_background(row.cells[1], "F8FAFC")
        set_cell_margins(row.cells[0], top=60, bottom=60, left=100, right=100)
        set_cell_margins(row.cells[1], top=60, bottom=60, left=100, right=100)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # -------------------------------------------------------------
    # SECTION 1: EXECUTIVE PROJECT SUMMARY
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary & Problem Statement", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(4)
    
    p = doc.add_paragraph()
    p.add_run(
        "Industrial construction sites, chemical plants, manufacturing facilities, and oil rigs enforce strict "
        "Personal Protective Equipment (PPE) regulations (such as OSHA 1926.100 for Head Protection and OSHA 1926.201 for "
        "High-Visibility Safety Vests). Manual safety inspections are sporadic, labor-intensive, error-prone, and impossible "
        "to sustain continuously across large facilities."
    )
    
    p2 = doc.add_paragraph()
    p2.add_run(
        "This project presents an automated, end-to-end, edge-deployable real-time safety compliance platform. "
        "Rather than treating safety gear detection as a naive bounding box exercise, this system solves the four critical "
        "engineering challenges in visual compliance:"
    )
    
    bullets = [
        ("Person-to-Equipment Association", "Generic detectors find 'hard hat' and 'person' separately. Our deterministic geometric engine maps which safety gear physically belongs to which worker using anatomical body proportions."),
        ("Biometric Identity Verification", "Privacy-preserving facial feature extraction (YuNet + SFace) matches registered personnel against a local 128-D vector catalog without storing raw biometric photos."),
        ("Temporal Anti-Flicker Protection", "Sliding-window temporal smoothing (10-frame buffer with 60% presence threshold) prevents momentary head turns or motion blurs from triggering false compliance alarms."),
        ("High-Throughput Full-Stack Telemetry", "Asynchronous FastAPI backend isolates neural network inference in a dedicated background worker thread, streaming lightweight structured JSON telemetry over WebSockets to a sleek React industrial dashboard at 115+ FPS."),
    ]
    for b_title, b_desc in bullets:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(3)
        r_b = bp.add_run(f"{b_title}: ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        bp.add_run(b_desc)
        
    add_callout(
        doc,
        "During your demonstration, emphasize that this is NOT just a basic YOLO wrapper. It is a complete 6-stage analytical "
        "pipeline that unifies detection, multi-object tracking, anatomical spatial priors, biometric face recognition, "
        "temporal anti-flicker filtering, and an industrial-grade WebSocket dashboard.",
        title="DEMO PITCH HIGHLIGHT"
    )

    # -------------------------------------------------------------
    # SECTION 2: ARCHITECTURE & HOW THE SYSTEM WORKS
    # -------------------------------------------------------------
    h1 = doc.add_heading("2. System Architecture & End-to-End Data Flow", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    
    p = doc.add_paragraph()
    p.add_run(
        "The system processes live camera feeds or prerecorded video through an 8-stage synchronous perception pipeline, "
        "decoupled from network clients via an asynchronous event dispatcher:"
    )
    
    # Pipeline stages table
    pipe_table = doc.add_table(rows=1, cols=4)
    pipe_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    pipe_widths = [Inches(0.8), Inches(1.8), Inches(2.2), Inches(1.7)]
    pipe_headers = ["Stage", "Component Module", "Transformation / Operation", "Output / Format"]
    format_table_headers(pipe_table, pipe_widths, pipe_headers)
    
    stages = [
        ("1", "Video Ingestion (OpenCV)", "Captures frames from webcam, RTSP, or video file at native resolution", "RGB/BGR uint8 array (640x480)"),
        ("2", "YOLOv8n Detector (detector.py)", "Runs FP16 Tensor Core inference on GPU to locate Persons and PPE classes", "List of Detection objects with xyxy, class_id, conf"),
        ("3", "IoU Tracker (pipeline.py)", "Greedy bipartite matching with existing tracks (IoU >= 0.30, max_missed=15)", "ActiveTrack instances with track_id ('track_1')"),
        ("4", "Spatial Associator (ppe_association.py)", "Evaluates anatomical head/torso priors, containment ratio & vertical midline proximity", "AssociatedPerson mapping gear to specific person"),
        ("5", "Face Biometrics (embeddings.py)", "Crops head region, locates 5 landmarks via YuNet, extracts 128-D vector via SFace", "Employee ID match (cosine >= 0.363) or UNKNOWN"),
        ("6", "Temporal Filter (temporal_filter.py)", "Sliding window of last 10 frames per track with 60% voting presence threshold", "SmoothedPersonState with stabilized boolean states"),
        ("7", "Compliance Engine (rules.py)", "Pure Python deterministic rule evaluation: requires Hardhat AND Safety Vest", "COMPLIANT or NON_COMPLIANT + missing items"),
        ("8", "WebSocket Dispatch (FastAPI)", "Serializes consolidated records to structured JSON and broadcasts to clients", "Lightweight JSON payload (NO heavy raw video)"),
    ]
    for s_idx, row_data in enumerate(stages):
        row = pipe_table.add_row()
        style_data_row(row, pipe_widths, row_data, is_even=(s_idx % 2 == 1))
        
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # SECTION 3: TECHNOLOGIES, MODELS & TOOLS USED
    # -------------------------------------------------------------
    h1 = doc.add_heading("3. Technologies, Frameworks & Models Used", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    
    p = doc.add_paragraph()
    p.add_run(
        "Every technology chosen in this capstone was selected to maximize real-time inference throughput, "
        "minimize system latency, eliminate fragile external dependencies, and preserve user privacy:"
    )
    
    tech_table = doc.add_table(rows=1, cols=3)
    tech_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tech_widths = [Inches(1.8), Inches(2.3), Inches(2.4)]
    tech_headers = ["Layer / Subsystem", "Technology / Model Employed", "Technical Rationale & Advantage"]
    format_table_headers(tech_table, tech_widths, tech_headers)
    
    tech_stack = [
        ("Object Detection Model", "Ultralytics YOLOv8n (Nano)\n- PyTorch 2.6.0+cu124\n- ONNX Runtime & ONNXSlim", "Only 3.0M parameters and 8.1 GFLOPs. Executes in 7.4 ms on RTX 4050 (135 FPS), leaving abundant GPU headroom for other pipeline stages."),
        ("Face Detection Model", "OpenCV YuNet (CNN)\n(face_detection_yunet_2023mar.onnx)", "Lightweight 232 KB model that outputs precise 5 facial landmarks (eyes, nose, mouth corners) even with partial occlusion and hardhat brims."),
        ("Face Recognition Model", "OpenCV SFace (SphereFace)\n(face_recognition_sface_2021dec.onnx)", "Official OpenCV DNN deep recognizer (38.7 MB) extracting 128-D L2-normalized float32 vectors. Runs in 1.1 ms without dlib/CMake compile issues."),
        ("Multi-Object Tracking", "Custom SimpleIoUTracker\n- Bipartite greedy IoU matching", "Spatial tracking without appearance embedding overhead (runs at >10,000 FPS), maintaining track IDs across temporary frame drops."),
        ("Temporal Smoothing", "Custom PPETemporalFilter\n- collections.deque circular buffers", "Sub-millisecond sliding window voting filter that eliminates single-frame false alerts and detection flickering."),
        ("Backend Framework", "FastAPI + Uvicorn + Pydantic v2\n- Standard Python threading", "Dedicated daemon worker thread isolates ML execution so heavy computer vision processing never blocks the asynchronous WebSocket event loop."),
        ("Frontend Dashboard", "React 19 + Vite\n- HTML5 Canvas 2D Rendering\n- Lucide React Icons", "Industrial dark theme with OSHA/ISO color coding. Renders bounding boxes and telemetry overlays smoothly on HTML5 canvas using dynamic coordinate scaling."),
        ("Automated Quality", "Pytest Test Suite\n- 177 unit & integration tests", "Guarantees mathematical correctness of geometric association, temporal voting edge cases, REST API schemas, and WebSocket streaming."),
    ]
    for t_idx, row_data in enumerate(tech_stack):
        row = tech_table.add_row()
        style_data_row(row, tech_widths, row_data, is_even=(t_idx % 2 == 1))

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # SECTION 4: HOW THINGS ARE IMPLEMENTED (DEEP DIVE)
    # -------------------------------------------------------------
    h1 = doc.add_heading("4. Detailed Technical Implementation", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)

    # 4.1 YOLO
    h2 = doc.add_heading("4.1. YOLOv8n Object Detection & Optimization", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(2)
    
    p = doc.add_paragraph()
    p.add_run(
        "The perception engine utilizes a fine-tuned Ultralytics YOLOv8n detector. We conducted an empirical investigation "
        "comparing a 25-class baseline model (trained on all raw construction site labels including background dump trucks and excavators) "
        "against an optimized 5-class target model focused exclusively on compliance entities: "
    )
    p.add_run("Person, Hardhat, NO-Hardhat, Safety Vest, and NO-Safety Vest.").bold = True
    
    p = doc.add_paragraph()
    p.add_run(
        "By pruning 20 extraneous background classes, backpropagation gradients were concentrated entirely on human and PPE feature "
        "representations. This yielded substantial empirical gains on identical test sets:"
    )

    eval_table = doc.add_table(rows=1, cols=5)
    eval_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    eval_widths = [Inches(1.8), Inches(1.1), Inches(1.2), Inches(1.2), Inches(1.2)]
    eval_headers = ["Metric (Test Set)", "Baseline (25-Class)", "Clean 5-Class", "Empirical Delta", "Relative Gain"]
    format_table_headers(eval_table, eval_widths, eval_headers)
    
    eval_rows = [
        ("mAP@50", "52.41%", "61.53%", "+9.12%", "+17.4% Relative"),
        ("mAP@50-95", "36.86%", "38.19%", "+1.33%", "+3.6% Relative"),
        ("Overall Precision", "69.71%", "78.63%", "+8.92%", "+12.8% Relative"),
        ("Overall Recall", "48.71%", "58.03%", "+9.32%", "+19.1% Relative"),
        ("Hardhat Precision", "87.27%", "94.47%", "+7.20%", "+8.3% Relative"),
    ]
    for r_idx, row_data in enumerate(eval_rows):
        row = eval_table.add_row()
        style_data_row(row, eval_widths, row_data, is_even=(r_idx % 2 == 1))
        
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    add_callout(
        doc,
        "Crucial defense point: Hardhat precision is 94.47%. In industrial safety, high precision on safety equipment prevents "
        "deadly false compliance passes. FP16 half-precision on NVIDIA Tensor Cores cuts detector latency down to 7.41 ms (134.9 FPS).",
        title="VIVA HIGHLIGHT"
    )

    # 4.2 Person-to-PPE Spatial Association
    h2 = doc.add_heading("4.2. Person-to-PPE Spatial Association (Anatomical Priors)", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(2)
    
    p = doc.add_paragraph()
    p.add_run(
        "A standard object detector only generates independent bounding boxes. It does not know if a hardhat belongs to Person A "
        "or Person B. To resolve this without running an expensive second neural network, we designed a deterministic "
        "geometric association algorithm based on human anatomical proportions (detection/ppe_association.py):"
    )
    
    bullets = [
        ("Relative Coordinates", "Given a person box B_p = (x_p1, y_p1, x_p2, y_p2) of height H_p and PPE center (cx, cy), relative vertical position is rel_y = (cy - y_p1) / H_p and rel_x = (cx - x_p1) / W_p."),
        ("Head Region Prior", "Candidate headgear (Hardhat, NO-Hardhat, Mask) must lie within rel_y in [-0.20, 0.38] with ideal anchor at 0.08. The negative upper bound (-0.20) accounts for hardhats protruding above the person's detected bounding box."),
        ("Torso Region Prior", "Candidate torso items (Safety Vest, NO-Safety Vest) must lie within rel_y in [0.12, 0.72] with ideal anchor at 0.38."),
        ("Multi-Factor Scoring Function", "Score S = 0.45 * Containment + 0.40 * Proximity + 0.15 * Confidence. Containment is Area(B_p ∩ B_ppe) / Area(B_ppe). Proximity penalizes horizontal deviation from the vertical spine midline."),
        ("Exclusive Assignment", "If multiple workers share candidate space, the PPE item is assigned exclusively to the person achieving the highest score S. Spare equipment on tables or ground fails min_score (0.20) and is cleanly unassigned."),
    ]
    for b_title, b_desc in bullets:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(3)
        r_b = bp.add_run(f"{b_title}: ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        bp.add_run(b_desc)

    # 4.3 Face Biometrics
    h2 = doc.add_heading("4.3. Privacy-Preserving Biometric Face Recognition", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(2)
    
    p = doc.add_paragraph()
    p.add_run(
        "The biometric pipeline identifies registered workers using official OpenCV Zoo deep learning models "
        "(face_detection_yunet_2023mar.onnx and face_recognition_sface_2021dec.onnx):"
    )
    
    face_steps = [
        ("1. Multi-Sample Enrollment", "Workers register by providing 1 to 5 face photos. YuNet detects the face and extracts 5 facial landmarks. If 0 faces or >1 faces are present in a photo, the photo is immediately rejected to prevent corrupted data."),
        ("2. Geometric Alignment & Affine Normalization", "SFace AlignCrop uses the 5 landmarks to rotate and scale the facial region into a normalized 112x112 frontal crop, compensating for head tilts."),
        ("3. 128-D Feature Embedding", "The SFace deep convolutional network maps the aligned face into a 128-dimensional L2-normalized float32 vector on a unit hypersphere."),
        ("4. Strict Cosine Verification", "Query faces are matched against registered employees using Cosine Similarity. If the maximum similarity is below 0.363 (official SFace threshold), the identity is strictly kept as 'UNKNOWN'. There is zero forced identity guessing."),
        ("5. Privacy-by-Design Architecture", "Raw face photographs are processed entirely in volatile RAM and immediately discarded. Only mathematical embedding arrays (embeddings.npz, ~512 bytes per vector) and metadata are stored on disk."),
    ]
    for f_title, f_desc in face_steps:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(3)
        r_b = bp.add_run(f"{f_title}: ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        bp.add_run(f_desc)

    # 4.4 Temporal Smoothing Filter
    h2 = doc.add_heading("4.4. Sliding-Window Temporal Smoothing Filter", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(2)
    
    p = doc.add_paragraph()
    p.add_run(
        "Single-frame vision systems are prone to high-frequency flickering: an individual wearing a hardhat may momentarily "
        "tilt their head down, causing the detector to miss the hardhat for 1 or 2 frames. Without filtering, this triggers "
        "jarring alarms and alert fatigue. Our temporal filter (compliance/temporal_filter.py) solves this:"
    )
    
    filter_points = [
        ("Sliding Window Deque", "Maintains a fixed-capacity circular buffer (default: window_size = 10 frames) for each active track ID."),
        ("Fractional Voting", "An item is declared stably PRESENT if observed in >= 60% of the window frames (i.e. >= 6 out of 10 frames)."),
        ("Asymmetrical Responsiveness", "Can tolerate up to 4 consecutive missed frames due to motion blur or occlusion without switching status to NON_COMPLIANT."),
        ("Memory & Speed", "Implemented with Python collections.deque, processing 10,000+ tracks per second with negligible 0.01 ms latency."),
    ]
    for fp_title, fp_desc in filter_points:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(3)
        r_b = bp.add_run(f"{fp_title}: ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        bp.add_run(fp_desc)

    # 4.5 Backend & Frontend
    h2 = doc.add_heading("4.5. Backend Concurrency & Frontend Dashboard Architecture", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(2)
    
    p = doc.add_paragraph()
    p.add_run(
        "Industrial monitoring demands rock-solid backend reliability and responsive telemetry visualization:"
    )
    
    arch_points = [
        ("Inference Thread Isolation", "Computer vision inference (OpenCV capture, YOLO forward pass, SFace matching) runs inside a dedicated background daemon thread (MonitoringService._worker_loop). This guarantees that CPU/GPU-intensive inference never blocks the asyncio event loop handling HTTP requests and WebSocket connections."),
        ("Bandwidth-Efficient Telemetry", "Unlike naive systems that stream heavy MJPEG video streams over WebSockets (consuming 50-100 Mbps), our backend sends only structured JSON telemetry packets (~2 KB per update containing track IDs, normalized bboxes, compliance booleans, and latency metrics)."),
        ("Dynamic Canvas Coordinate Projection", "The React frontend receives bboxes in standard 640x480 coordinate space. An HTML5 Canvas hook dynamically calculates horizontal and vertical scale factors based on browser viewport size, drawing crisp target reticles, bounding boxes, worker name tags, and compliance badges at 60 FPS."),
        ("Interactive Management", "The dashboard provides live start/stop stream toggles, horizontal camera mirroring, real-time KPI metrics, employee registration with live image preview, and an enrolled personnel directory."),
    ]
    for ap_title, ap_desc in arch_points:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(3)
        r_b = bp.add_run(f"{ap_title}: ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        bp.add_run(ap_desc)

    # -------------------------------------------------------------
    # SECTION 5: LIVE DEMONSTRATION WALKTHROUGH SCRIPT
    # -------------------------------------------------------------
    h1 = doc.add_heading("5. Step-by-Step Live Demonstration Script", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    
    p = doc.add_paragraph()
    p.add_run(
        "Follow this exact sequential demonstration script during your presentation tomorrow to showcase every aspect "
        "of the platform smoothly and professionally:"
    )

    demo_steps = [
        ("Step 1: One-Click System Launch",
         "Action: In the terminal, execute .\\start.bat (or double click start.bat).\n"
         "What Happens: Validates .venv and frontend node_modules, boots FastAPI on port 8000, starts Vite on port 5173, and launches the browser automatically.\n"
         "Speaking Point: 'We designed the deployment for zero-friction industrial operations. A single script validates GPU CUDA bindings, initializes the backend, and boots the real-time React dashboard.'"),
        
        ("Step 2: Dashboard Overview & UI Aesthetics",
         "Action: Direct attention to the live dashboard at http://localhost:5173/dashboard.\n"
         "Speaking Point: 'The dashboard uses an industrial dark theme aligned with OSHA compliance standards. The top KPI cards show total tracked personnel, compliant workers, and active violations. The center viewport renders the live spatial perception grid with real-time hardware telemetry.'"),
        
        ("Step 3: Start Live Video Feed & Real-Time Telemetry",
         "Action: Click 'Start Camera' (or 'Start Monitoring' with camera 0 / video source).\n"
         "What to Show: Step in front of the camera (or point webcam at a test image/subject). Notice the tracking reticle, green/red bounding box, and the latency telemetry badge in the top right.\n"
         "Speaking Point: 'Notice the latency breakdown: YOLOv8 detection runs in ~7.4 ms, spatial association in 0.01 ms, and temporal smoothing in 0.01 ms. Total end-to-end throughput exceeds 110 frames per second on this RTX 4050 GPU.'"),
         
        ("Step 4: Demonstrate Safety Compliance & Violation Alerts",
         "Action: \n"
         "- Show compliance: Wear a hardhat/cap and high-visibility vest (or hold up picture). The box turns GREEN ('COMPLIANT').\n"
         "- Show violation: Take off the hardhat. The box immediately turns RED ('NON_COMPLIANT'), showing 'Missing: Hardhat', and an active violation alert appears on the banner.\n"
         "Speaking Point: 'Our deterministic compliance engine enforces OSHA requirements. When any required safety item is missing, the system instantly flags a violation without requiring cloud round-trips.'"),
         
        ("Step 5: Demonstrate Temporal Anti-Flicker Protection",
         "Action: Quickly turn your head away for half a second or briefly wave a hand in front of the hardhat.\n"
         "What to Show: Notice the box does NOT immediately flicker into red or spam alerts.\n"
         "Speaking Point: 'Notice how stable the system is. In traditional detection systems, a single blurry frame causes alert flickering. Our sliding-window temporal filter requires a 60% presence consensus over 10 frames, absorbing optical dropouts.'"),
         
        ("Step 6: Biometric Employee Registration & Identification",
         "Action: Navigate to http://localhost:5173/register in the navbar. Enter an Employee ID (e.g., EMP101) and Name (e.g., 'Alex Vance'). Drag and drop 2 portrait photos. Click 'Register Employee'.\n"
         "Then return to Dashboard / Live Feed.\n"
         "What to Show: The person bounding box updates from 'UNKNOWN' to 'EMP101: Alex Vance' with a cosine similarity score.\n"
         "Speaking Point: 'Here we demonstrate our privacy-preserving biometric enrollment. YuNet aligned the face landmarks, SFace computed a 128-D vector, and raw photos were immediately purged from memory. Only mathematical embeddings are stored in catalog.json and embeddings.npz.'"),
         
        ("Step 7: Interactive API Documentation & Automated Tests",
         "Action: \n"
         "1. Open http://localhost:8000/docs in a tab to show Swagger OpenAPI.\n"
         "2. In terminal, run: .venv\\Scripts\\python.exe -m pytest tests/ -v\n"
         "Speaking Point: 'The entire system is backed by a fully documented OpenAPI specification and 177 automated pytest tests covering geometric association math, temporal smoothing edge cases, and WebSocket streams with 100% test pass rate.'"),
    ]
    
    for s_title, s_desc in demo_steps:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        r_t = p.add_run(s_title)
        r_t.bold = True
        r_t.font.size = Pt(11.5)
        r_t.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
        
        p_desc = doc.add_paragraph()
        p_desc.paragraph_format.space_before = Pt(0)
        p_desc.paragraph_format.space_after = Pt(4)
        for line in s_desc.split("\n"):
            if line.startswith("Action:"):
                r = p_desc.add_run("Action: ")
                r.bold = True
                p_desc.add_run(line.replace("Action:", "").strip() + "\n")
            elif line.startswith("What Happens:") or line.startswith("What to Show:"):
                prefix = "What to Show: " if "Show" in line else "What Happens: "
                r = p_desc.add_run(prefix)
                r.bold = True
                p_desc.add_run(line.split(":", 1)[1].strip() + "\n")
            elif line.startswith("Speaking Point:"):
                r = p_desc.add_run("What to Say: ")
                r.bold = True
                r.font.color.rgb = RGBColor(0x05, 0x96, 0x69) # Emerald 600
                p_desc.add_run(line.replace("Speaking Point:", "").strip() + "\n")
            else:
                p_desc.add_run(line + "\n")

    # -------------------------------------------------------------
    # SECTION 6: VIVA DEFENSE QUESTIONS & ANSWERS
    # -------------------------------------------------------------
    h1 = doc.add_heading("6. Evaluator & Viva Defense Questions (With Model Answers)", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    
    p = doc.add_paragraph()
    p.add_run(
        "Below are the 12 most critical technical questions evaluators and professors are likely to ask during your demonstration, "
        "along with authoritative model answers based directly on your codebase:"
    )

    viva_qa = [
        ("Q1: Why did you use YOLOv8n (Nano) instead of larger variants like YOLOv8x or a two-stage detector like Faster R-CNN?",
         "Model Answer: In industrial monitoring, edge latency and frame rate are paramount. Faster R-CNN typically runs at 15-25 FPS with 150+ ms latency on edge hardware. YOLOv8n has only 3.0 million parameters and 8.1 GFLOPs, executing in 7.4 ms (over 130 FPS) with FP16 Tensor Cores. Furthermore, our empirical fine-tuning on the clean 5-class PPE dataset achieved 94.5% Hardhat precision and 61.5% mAP@50, proving that a lightweight architecture provides the necessary accuracy while leaving 85% of GPU cycles free for multi-stream tracking and biometric inference."),
        
        ("Q2: How do you solve the Person-to-PPE association problem if two workers stand close together?",
         "Model Answer: Standard object detectors output isolated bounding boxes without linking gear to people. We implemented a deterministic geometric association algorithm in detection/ppe_association.py. First, we apply anatomical vertical bounds: headgear must fall in relative height [-0.20, 0.38] and vests in [0.12, 0.72]. When multiple candidates overlap, we compute a composite score: S = 0.45*Containment + 0.40*Proximity + 0.15*Confidence. Proximity measures distance to the vertical spine midline. The worker with the highest score is assigned the PPE exclusively (1-to-1 matching). Any stray PPE failing a 0.20 score threshold remains unassigned."),
        
        ("Q3: Why not train an end-to-end multi-task neural network that predicts person, gear, and identity all in one shot?",
         "Model Answer: Monolithic multi-task networks suffer from negative gradient transfer (gradient interference), catastrophic forgetting, and immense retraining overhead. If a new safety item (e.g. welding goggles) is introduced, an end-to-end network requires retraining everything from scratch. Our modular architecture separates perception (YOLOv8), geometry (Anatomical Associator), biometrics (YuNet/SFace), and compliance rules (pure Python). This follows the Single Responsibility Principle: each module can be unit-tested, swapped, or scaled independently."),
        
        ("Q4: Why did you choose OpenCV YuNet and SFace instead of dlib, FaceNet, or DeepFace?",
         "Model Answer: First, portability and stability: dlib requires C++ CMake compilation on Windows, which frequently breaks in production Python environments. Second, memory and latency: FaceNet/DeepFace are heavy PyTorch/TensorFlow models taking hundreds of milliseconds. OpenCV YuNet is only 232 KB and extracts 5 facial landmarks in under 1 ms. SFace is a 38.7 MB ONNX model that generates 128-D normalized embeddings in 1.1 ms. Both run natively via OpenCV's cv2.dnn without any external deep learning dependencies."),
        
        ("Q5: How does your system address data privacy and GDPR regulations regarding facial biometric storage?",
         "Model Answer: The system is designed with Privacy-by-Design principles. Raw facial photographs are processed purely in volatile memory during registration and immediately garbage-collected. We never store image files or raw pixels on disk. Only 128-dimensional mathematical float32 unit vectors are saved in compressed .npz files (~512 bytes). Because deep face embeddings are non-invertible, original facial photos cannot be reconstructed from the stored vectors."),
        
        ("Q6: What prevents false alarms when a worker turns their head or bends over for a second?",
         "Model Answer: That is handled by our Temporal Smoothing Filter in compliance/temporal_filter.py. The filter maintains a sliding history window (collections.deque) of the last 10 frames for each active track ID. An item is marked present only if detected in >= 60% of the window. If a worker turns their head and the hardhat drops for 1-3 frames, the voting presence remains above 60%, preventing false alarm flickers. Only sustained non-compliance triggers an alert."),
        
        ("Q7: Why do you stream structured JSON over WebSockets instead of an MJPEG video stream?",
         "Model Answer: Streaming raw video or MJPEG frames from the backend to multiple browser clients causes severe network bandwidth saturation (50-100 Mbps for HD video) and CPU encoding bottlenecks. Our architecture transmits only lightweight structured JSON telemetry (~2-5 KB/s per client) containing worker coordinates, compliance booleans, and latency metrics. The frontend React client renders the canvas overlays locally at 60 FPS. In production, edge cameras decode RTSP feeds locally, keeping server bandwidth minimal."),
        
        ("Q8: Why did your 5-class candidate model outperform the 25-class baseline model by +9.1% mAP?",
         "Model Answer: The raw 25-class dataset contained severe class imbalance with rare background machinery classes (dump trucks, vans, trailers) that had fewer than 5 instances. During backpropagation, these rare classes caused gradient dilution and false cross-class penalties against workers. By pruning the dataset to the 5 core safety classes, gradient updates focused entirely on distinguishing humans, hardhats, and safety vests, increasing mAP@50 from 52.41% to 61.53% and recall from 48.71% to 58.03%."),
        
        ("Q9: What happens if a worker leaves the camera frame and comes back later?",
         "Model Answer: Our SimpleIoUTracker maintains active tracks with a max_missed=15 frame grace period. If a worker briefly steps behind a column and reappears within half a second, their track ID is preserved. If they exit for longer, the dead track is pruned. When they return, a new track ID is spawned, and the biometric face engine re-identifies them as soon as their face is visible, re-associating their employee record automatically."),
        
        ("Q10: How does the backend ensure that heavy computer vision inference does not freeze API requests?",
         "Model Answer: In backend/services/monitoring_service.py, camera capture and the entire ML pipeline run inside a dedicated background daemon thread. The inference thread writes frame summaries into a thread-safe cache and puts JSON events into an asyncio queue. The FastAPI ASGI event loop remains completely unblocked to handle incoming HTTP requests and WebSocket client handshakes with sub-millisecond response times."),
        
        ("Q11: What are the primary operating boundaries and limitations of this system?",
         "Model Answer: As documented in docs/known_limitations.md: 1) Scale: workers must occupy at least 80x40 pixels for PPE association and 40x40 pixels for face detection; 2) Occlusion: extreme crouching shifts vertical priors, which we plan to address using 2D pose keypoints (YOLOv8-pose); 3) Facial yaw: face recognition requires facial orientation within ±45 degrees yaw and ±30 degrees pitch. Crucially, when face recognition fails, the system safely marks the identity as UNKNOWN rather than making a false guess."),
        
        ("Q12: How did you test and validate the system for correctness?",
         "Model Answer: We built a comprehensive automated test suite with 177 unit and integration tests using pytest. It covers pure Python compliance rules, boundary conditions for anatomical spatial priors, sliding-window voting edge cases, biometric enrollment rejection logic (0 faces or multiple faces), FastAPI REST endpoints, and WebSocket event payloads. All 177 tests execute in 2.3 seconds with 100% pass rate."),
    ]

    for q_text, a_text in viva_qa:
        qp = doc.add_paragraph()
        qp.paragraph_format.space_before = Pt(8)
        qp.paragraph_format.space_after = Pt(2)
        r_q = qp.add_run(q_text)
        r_q.bold = True
        r_q.font.size = Pt(11)
        r_q.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        
        ap = doc.add_paragraph()
        ap.paragraph_format.space_before = Pt(0)
        ap.paragraph_format.space_after = Pt(4)
        r_a_title = ap.add_run("Answer: ")
        r_a_title.bold = True
        r_a_title.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
        ap.add_run(a_text.replace("Model Answer: ", ""))

    # -------------------------------------------------------------
    # SECTION 7: BENCHMARKS & HARDWARE METRICS
    # -------------------------------------------------------------
    h1 = doc.add_heading("7. Latency, Throughput & Hardware Performance", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    
    p = doc.add_paragraph()
    p.add_run(
        "Empirical benchmarks measured over 100 iterations on an NVIDIA GeForce RTX 4050 Laptop GPU (640x480 resolution):"
    )

    bench_table = doc.add_table(rows=1, cols=5)
    bench_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    bench_widths = [Inches(2.0), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.2)]
    bench_headers = ["Pipeline Subsystem", "FP32 Latency", "FP16 Latency", "FP16 P95", "Throughput (FPS)"]
    format_table_headers(bench_table, bench_widths, bench_headers)
    
    bench_rows = [
        ("YOLOv8 Detection", "8.37 ms", "7.41 ms", "10.79 ms", "134.9 FPS"),
        ("Spatial PPE Association", "0.01 ms", "0.01 ms", "0.01 ms", ">100,000 FPS"),
        ("Face Feature Extraction (SFace)", "1.16 ms", "1.08 ms", "1.29 ms", "926.7 FPS"),
        ("Temporal Smoothing Filter", "0.01 ms", "0.01 ms", "0.01 ms", ">80,000 FPS"),
        ("Unified End-to-End Pipeline", "9.23 ms", "8.69 ms", "12.76 ms", "115.0 FPS"),
    ]
    for b_idx, row_data in enumerate(bench_rows):
        row = bench_table.add_row()
        style_data_row(row, bench_widths, row_data, is_even=(b_idx % 2 == 1))
        
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # Save document
    out_path = Path("d:/ppe-compliance/PPE_Compliance_Project_Demonstration_Guide.docx")
    doc.save(str(out_path))
    print(f"Successfully generated demonstration guide at: {out_path}")

if __name__ == "__main__":
    build_demo_guide()
