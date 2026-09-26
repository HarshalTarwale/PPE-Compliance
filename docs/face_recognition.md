# Employee Face Registration & Recognition Architecture

## 1. Executive Summary

The Face Recognition and Registration subsystem provides privacy-preserving identity verification for workers on site. It pairs with the PPE detection system to attribute compliance or violation records to registered individuals without storing sensitive raw biometric photos.

The system is implemented in:
- [`recognition/embeddings.py`](file:///d:/ppe-compliance/recognition/embeddings.py): Face detection, 5-point landmark alignment, and 128-D feature extraction.
- [`recognition/registration.py`](file:///d:/ppe-compliance/recognition/registration.py): Multi-image enrollment, validation/rejection of ambiguous photos, local embedding persistence, and threshold-based identification.

---

## 2. Enrollment Pipeline

```
  Enrollment Photos (Multiple Images)
                 │
                 ▼
  ┌────────────────────────────────────────────────────────┐
  │  1. Face Detection (YuNet ONNX)                        │
  │     - Extracts bounding box & 5 landmarks              │
  │     - If 0 faces: REJECT (NO_FACE_DETECTED)            │
  │     - If >1 faces: REJECT (MULTIPLE_FACES_DETECTED)    │
  └────────────────────────────────────────────────────────┘
                 │ (Exactly 1 Face Found)
                 ▼
  ┌────────────────────────────────────────────────────────┐
  │  2. Landmark Alignment & Cropping (SFace AlignCrop)    │
  │     - Affine rotation & normalization to 112x112       │
  └────────────────────────────────────────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────────────────────────┐
  │  3. Feature Extraction (SFace Deep CNN)                │
  │     - Computes 128-D L2-normalized vector per image    │
  └────────────────────────────────────────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────────────────────────┐
  │  4. Aggregate Representation                           │
  │     - v_mean = normalize(mean(v_1, v_2, ..., v_k))     │
  └────────────────────────────────────────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────────────────────────┐
  │  5. Privacy-Preserving Local Storage                   │
  │     - Discards raw images immediately                  │
  │     - Stores record.json + embeddings.npz              │
  └────────────────────────────────────────────────────────┘
```

---

## 3. Installation & Dependency Requirements

The face recognition subsystem was designed specifically to avoid fragile C++ compilation dependencies (such as `dlib` or `CMake` on Windows) and version conflicts with PyTorch.

### Dependencies
- **Python**: 3.10+ / 3.11
- **OpenCV**: `opencv-python>=4.8.0` (utilizes the built-in `cv2.dnn` module)
- **NumPy**: `numpy>=1.24.0`

No additional external deep learning frameworks are required for face recognition.

---

## 4. Model Requirements

The subsystem uses official deep learning models published by the OpenCV Zoo:

| Component | Model Name | Architecture | File Size | Output |
| :--- | :--- | :--- | :---: | :--- |
| **Face Detector** | `face_detection_yunet_2023mar.onnx` | YuNet (CNN) | `232 KB` | Bounding box + 5 landmarks (eyes, nose, mouth corners) |
| **Face Recognizer** | `face_recognition_sface_2021dec.onnx` | SFace (SphereFace / Tsinghua) | `38.7 MB` | 128-dimensional L2-normalized float32 embedding vector |

### Model Storage Location
Models are stored under:
```
models/face/
├── face_detection_yunet_2023mar.onnx
└── face_recognition_sface_2021dec.onnx
```

---

## 5. Privacy-Preserving Storage Architecture

Raw biometric facial images are **never stored** on disk:
- **`data/employees/catalog.json`**: Global index of registered employee IDs, names, registration timestamps, and sample counts.
- **`data/employees/<employee_id>/record.json`**: Individual employee metadata.
- **`data/employees/<employee_id>/embeddings.npz`**: Compressed NumPy archive storing only the 128-D `mean_embedding` and individual `sample_embeddings` (each embedding is 128 floats $\approx$ 512 bytes).

---

## 6. Ambiguity Handling & Strict Verification

1. **Zero Faces Detected**:
   - If an enrollment photo does not clearly show a face, it is rejected with status `NO_FACE_DETECTED`.
2. **Multiple Faces Detected**:
   - If an enrollment photo contains more than one face (e.g. a bystander in the background), it is rejected with status `MULTIPLE_FACES_DETECTED` to prevent identity contamination.
3. **No Automatic Guessing of Unknown Persons**:
   - When identifying an individual, the cosine similarity between the query face and registered employees must exceed the configurable threshold:
     $$\text{Cosine Similarity} = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} \ge \text{threshold}$$
   - Default threshold: **`0.363`** (OpenCV SFace standard).
   - If no registered employee reaches the threshold, the service strictly returns `None` (`Unknown`).

---

## 7. Limitations & Operational Guidelines

| Limitation | Impact | Recommended Mitigation |
| :--- | :--- | :--- |
| **Mask / Lower Face Occlusion** | Masks cover the nose and mouth landmarks, reducing recognition accuracy. | Enroll workers without masks during initial registration. During live inference, identify workers before they don face masks. |
| **Hardhat Brim Shadow** | Strong overhead lighting can cast deep shadows over the eyes and forehead. | Enroll multiple photos under varied lighting and angles (slight tilt, straight-on). |
| **Extreme Poses ($> 45^\circ$)** | Face detectors fail when the face is turned away from the camera. | Require frontal or near-frontal orientation during check-in / checkpoint monitoring. |
| **Low Resolution** | Faces smaller than $30 \times 30$ pixels lack feature detail. | Ensure monitoring cameras have adequate optical resolution at checkpoints. |
