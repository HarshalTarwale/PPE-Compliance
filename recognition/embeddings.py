"""
Face Embedding and Landmark Alignment Engine.

Provides deep-learning face detection, landmark-based alignment, and 128-dimensional
feature embedding extraction using OpenCV's official DNN YuNet and SFace models.

Completely independent of YOLO and detection components.

Pipeline:
  Input Image (BGR)
    │
    ▼
  Face Detection (YuNet)  ──► 5 Landmarks (Eyes, Nose, Mouth Corners)
    │
    ▼
  Face Alignment & Crop (SFace AlignCrop)  ──► Aligned 112x112 Face
    │
    ▼
  Feature Extraction (SFace Deep Recognizer)  ──► 128-D L2-Normalized Embedding
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is importable
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("embeddings")

# Default model locations
DEFAULT_YUNET_PATH = _PROJECT_ROOT / "models" / "face" / "face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_PATH = _PROJECT_ROOT / "models" / "face" / "face_recognition_sface_2021dec.onnx"

# Default cosine similarity threshold recommended by OpenCV SFace documentation
DEFAULT_COSINE_THRESHOLD: float = 0.363


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceDetection:
    """Represents a single detected face with bounding box and landmarks."""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    raw_array: np.ndarray  # 15-element array required by SFace alignCrop
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def box_xywh(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h)."""
        return self.x, self.y, self.w, self.h

    @property
    def box_xyxy(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2)."""
        return self.x, self.y, self.x + self.w, self.y + self.h

    @property
    def center(self) -> Tuple[float, float]:
        """Return center (cx, cy)."""
        return self.x + self.w / 2.0, self.y + self.h / 2.0


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------


class FaceEmbeddingEngine:
    """Face detection, alignment, and embedding extraction using OpenCV DNN."""

    def __init__(
        self,
        yunet_path: Optional[Union[str, Path]] = None,
        sface_path: Optional[Union[str, Path]] = None,
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    ) -> None:
        """Initialize face detector (YuNet) and recognizer (SFace).

        Args:
            yunet_path: Path to YuNet ONNX detection model.
            sface_path: Path to SFace ONNX recognition model.
            score_threshold: Minimum detection confidence for face detector.
            nms_threshold: Non-Maximum Suppression IoU threshold.
            cosine_threshold: Default similarity threshold for positive verification.
        """
        self.yunet_path = Path(yunet_path) if yunet_path else DEFAULT_YUNET_PATH
        self.sface_path = Path(sface_path) if sface_path else DEFAULT_SFACE_PATH
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.cosine_threshold = float(cosine_threshold)

        if not self.yunet_path.is_file():
            raise FileNotFoundError(
                f"Face detection model (YuNet) not found: {self.yunet_path.resolve()}"
            )
        if not self.sface_path.is_file():
            raise FileNotFoundError(
                f"Face recognition model (SFace) not found: {self.sface_path.resolve()}"
            )

        logger.info("Initializing FaceDetectorYN with: %s", self.yunet_path.name)
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.yunet_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

        logger.info("Initializing FaceRecognizerSF with: %s", self.sface_path.name)
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(self.sface_path),
            config="",
        )
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Detection & Alignment
    # -----------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect all faces in an image with 5-point facial landmarks.

        Args:
            image: Input image as BGR numpy array.

        Returns:
            List of FaceDetection objects sorted by confidence descending.
        """
        if image is None or image.size == 0:
            return []

        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        h, w = image.shape[:2]

        with self._lock:
            self.detector.setInputSize((w, h))
            try:
                _, faces = self.detector.detect(image)
            except cv2.error as err:
                logger.warning("FaceDetectorYN detect failed on image (%dx%d): %s", w, h, err)
                return []

        if faces is None or len(faces) == 0:
            return []

        detections: List[FaceDetection] = []
        for face_arr in faces:
            # YuNet face array structure:
            # [0:4] = x, y, w, h
            # [4:14] = landmarks: right_eye (4,5), left_eye (6,7), nose (8,9), right_mouth (10,11), left_mouth (12,13)
            # [14] = confidence score
            fx = max(0, int(round(face_arr[0])))
            fy = max(0, int(round(face_arr[1])))
            fw = max(1, int(round(face_arr[2])))
            fh = max(1, int(round(face_arr[3])))
            score = float(face_arr[14])

            landmarks = {
                "right_eye": (float(face_arr[4]), float(face_arr[5])),
                "left_eye": (float(face_arr[6]), float(face_arr[7])),
                "nose_tip": (float(face_arr[8]), float(face_arr[9])),
                "right_mouth": (float(face_arr[10]), float(face_arr[11])),
                "left_mouth": (float(face_arr[12]), float(face_arr[13])),
            }

            detections.append(
                FaceDetection(
                    x=fx,
                    y=fy,
                    w=fw,
                    h=fh,
                    confidence=score,
                    raw_array=face_arr.copy(),
                    landmarks=landmarks,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def align_and_crop(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        """Align face using 5-point landmarks and crop to standard 112x112 size.

        Args:
            image: Full original BGR image.
            face: FaceDetection containing raw 15-element landmark array.

        Returns:
            Aligned and cropped 112x112 BGR face image.
        """
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        with self._lock:
            aligned_face = self.recognizer.alignCrop(image, face.raw_array)
        return aligned_face

    def compute_embedding(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        """Extract a 128-dimensional L2-normalized feature embedding for a face.

        Args:
            image: Original full BGR image.
            face: FaceDetection object.

        Returns:
            1D numpy float32 array of shape (128,) with unit L2 norm.
        """
        aligned = self.align_and_crop(image, face)
        with self._lock:
            raw_feature = self.recognizer.feature(aligned)  # shape (1, 128)
        feature = raw_feature.flatten().astype(np.float32)

        # Ensure unit L2 norm
        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature = feature / norm

        return feature

    def extract_single_face_embedding(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], str, int]:
        """Strictly validate and extract embedding for registration.

        Requires EXACTLY 1 face in the image. Rejects images with 0 or >1 faces.

        Args:
            image: Input enrollment image.

        Returns:
            Tuple of (embedding, status_str, face_count):
              - If 1 face: (embedding_array, 'SUCCESS', 1)
              - If 0 faces: (None, 'NO_FACE_DETECTED', 0)
              - If >1 faces: (None, 'MULTIPLE_FACES_DETECTED', n)
        """
        if image is None or image.size == 0:
            return None, "NO_FACE_DETECTED", 0

        # Downscale oversized enrollment photos (e.g. 4000x3000 smartphone photos)
        # to max 1920px for reliable and fast DNN inference.
        h, w = image.shape[:2]
        max_dim = 1920
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        faces = self.detect_faces(image)
        count = len(faces)

        if count == 0:
            return None, "NO_FACE_DETECTED", 0
        if count > 1:
            return None, "MULTIPLE_FACES_DETECTED", count

        embedding = self.compute_embedding(image, faces[0])
        return embedding, "SUCCESS", 1

    # -----------------------------------------------------------------------
    # Similarity & Verification
    # -----------------------------------------------------------------------

    def compute_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two 128-dimensional face embeddings.

        Args:
            embedding_a: First 128-d embedding.
            embedding_b: Second 128-d embedding.

        Returns:
            Cosine similarity score in [-1.0, 1.0], where 1.0 indicates identical faces.
        """
        a = embedding_a.flatten()
        b = embedding_b.flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return round(similarity, 4)

    def is_match(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: Optional[float] = None,
    ) -> bool:
        """Determine whether two embeddings belong to the same person.

        Args:
            embedding_a: First embedding.
            embedding_b: Second embedding.
            threshold: Optional threshold override (defaults to self.cosine_threshold).

        Returns:
            True if similarity >= threshold, False otherwise.
        """
        thresh = threshold if threshold is not None else self.cosine_threshold
        sim = self.compute_similarity(embedding_a, embedding_b)
        return sim >= thresh
