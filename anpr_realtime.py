from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import logging
import os
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = ROOT_DIR
PADDLE_HOME_DIR = ROOT_DIR / "Paddle"
PADDLE_PDX_CACHE_DIR = ROOT_DIR / "PaddleX"
PADDLE_HOME_DIR.mkdir(parents=True, exist_ok=True)
PADDLE_PDX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("PADDLE_HOME", str(PADDLE_HOME_DIR))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_PDX_CACHE_DIR))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from ultralytics import YOLO

_PADDLEOCR_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
try:
    from paddleocr import PaddleOCR  # type: ignore
except ModuleNotFoundError as exc:
    PaddleOCR = Any  # type: ignore[assignment]
    _PADDLEOCR_IMPORT_ERROR = exc


DEFAULT_WEIGHTS = ROOT_DIR / "runs" / "detect" / "train2" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT_DIR / "WhatsApp Video 2026-03-14 at 12.58.39.mp4"
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BH_PLATE_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
READING_STATUS = "reading"
LOCKED_STATUS = "locked"
WEAK_LOCKED_STATUS = "weak_locked"
UNREADABLE_STATUS = "unreadable"
UNREADABLE_TEXT = "UNREADABLE"
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRPolicy:
    plate_size: str
    min_area: int
    min_width: int
    min_height: int
    min_aspect_ratio: float
    max_aspect_ratio: float
    scale_factor: float
    ocr_after_frames: int
    vote_threshold: int
    max_attempts: int
    min_confidence: float
    min_detection_confidence: float
    sharpness_threshold: float
    cooldown_frames: int
    lock_consensus: float


@dataclass
class AppConfig:
    weights: Path
    source: str
    conf_threshold: float = 0.35
    iou_threshold: float = 0.50
    imgsz: int = 640
    resize_width: int = 960
    tracker: str = "bytetrack"
    device: str = "cpu"
    output_path: Optional[Path] = None
    save_plates_dir: Optional[Path] = None
    log_file: Optional[Path] = None
    roi: Optional[str] = None
    frame_stride: int = 1
    min_frame_stride: int = 1
    max_frame_stride: int = 3
    adaptive_stride: bool = True
    stride_adjust_interval: int = 20
    target_fps_low: float = 20.0
    target_fps_high: float = 30.0
    max_fps: float = 0.0
    ocr_gate_confidence: float = 0.50
    ocr_min_width: int = 90
    ocr_min_height: int = 30
    ocr_min_aspect_ratio: float = 2.0
    ocr_max_aspect_ratio: float = 6.0
    ocr_min_confidence: float = 0.45
    min_plate_area: int = 2600
    tiny_plate_area: int = 280
    small_plate_area: int = 4200
    medium_plate_area: int = 11000
    small_plate_width: int = 150
    medium_plate_width: int = 240
    small_plate_scale: float = 3.0
    medium_plate_scale: float = 2.8
    large_plate_scale: float = 2.0
    large_plate_after_hits: int = 2
    medium_plate_after_hits: int = 2
    small_plate_after_hits: int = 2
    large_plate_vote_threshold: int = 2
    medium_plate_vote_threshold: int = 2
    small_plate_vote_threshold: int = 2
    large_plate_max_attempts: int = 6
    medium_plate_max_attempts: int = 6
    small_plate_max_attempts: int = 12
    large_plate_cooldown_frames: int = 1
    medium_plate_cooldown_frames: int = 2
    small_plate_cooldown_frames: int = 2
    ocr_min_interval: int = 2
    force_ocr_every_n_frames: int = 8
    blur_threshold: float = 30.0
    medium_plate_blur_threshold: float = 35.0
    small_plate_blur_threshold: float = 40.0
    max_plate_area_ratio: float = 0.05
    lock_consensus_large: float = 0.58
    lock_consensus_medium: float = 0.65
    lock_consensus_small: float = 0.72
    unreadable_after_age: int = 30
    track_ttl: int = 90
    bbox_history_size: int = 5
    ocr_history_size: int = 10
    ocr_consensus_ratio: float = 0.60
    enable_state_code_validation: bool = True
    use_clahe: bool = True
    use_sharpening: bool = True
    adaptive_threshold: bool = True
    display: bool = True
    use_half: bool = False


@dataclass
class TrackState:
    hits: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_ocr_frame: int = -10_000
    ocr_attempts: int = 0
    ocr_locked: bool = False
    status: str = READING_STATUS
    plate_text: str = ""
    ocr_confidence: float = 0.0
    consensus_ratio: float = 0.0
    crop_saved: bool = False
    logged: bool = False
    last_confidence: float = 0.0
    last_bbox: Optional[BBox] = None
    smoothed_bbox: Optional[BBox] = None
    plate_size: str = "unknown"
    last_blur_score: float = 0.0
    last_gate_reason: str = "new"
    last_vote_count: int = 0
    max_plate_area_seen: int = 0
    bbox_history: Deque[BBox] = field(default_factory=deque)
    ocr_votes: Counter[str] = field(default_factory=Counter)
    ocr_score_totals: Dict[str, float] = field(default_factory=dict)
    ocr_history: Deque[Tuple[int, str, float]] = field(default_factory=deque)


@dataclass
class PlateDetection:
    bbox: BBox
    track_id: int
    confidence: float
    plate_text: str = ""
    status: str = READING_STATUS
    reused: bool = False
    plate_size: str = "unknown"
    blur_score: float = 0.0
    ocr_confidence: float = 0.0
    consensus_ratio: float = 0.0
    gate_reason: str = ""
    vote_count: int = 0


@dataclass
class RuntimeState:
    track_states: Dict[int, TrackState] = field(default_factory=dict)
    plate_cache: Dict[int, str] = field(default_factory=dict)
    fps_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    total_frames: int = 0
    processed_frames: int = 0
    ocr_attempts_total: int = 0
    ocr_success_total: int = 0
    ocr_gated_total: int = 0
    ocr_strict_skip_total: int = 0
    current_frame_stride: int = 1
    last_stride_adjust_frame: int = 0
    last_roi_bbox: Optional[BBox] = None


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Production-style real-time ANPR with YOLOv8, ByteTrack, and PaddleOCR.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained YOLOv8 weights.")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="Video path or webcam index.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.50, help="Tracker IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size.")
    parser.add_argument("--resize-width", type=int, default=960, help="Resize frames to this width.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run detection every Nth frame.")
    parser.add_argument("--max-fps", type=float, default=0.0, help="Optional FPS cap. 0 disables cap.")
    parser.add_argument("--tracker", choices=["bytetrack"], default="bytetrack", help="Tracker backend.")
    parser.add_argument("--device", type=str, default=None, help="Inference device. Example: cpu, 0.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output video path.")
    parser.add_argument("--save-plates-dir", type=Path, default=None, help="Optional directory to save locked crops.")
    parser.add_argument("--log-file", type=Path, default=ROOT_DIR / "anpr_detections.log", help="Optional detection log file.")
    parser.add_argument("--roi", type=str, default=None, help="ROI as x1,y1,x2,y2 in pixels or normalized coordinates.")

    parser.add_argument("--ocr-after-hits", type=int, default=2, help="Legacy base OCR age threshold.")
    parser.add_argument("--ocr-vote-threshold", type=int, default=2, help="Legacy base vote threshold.")
    parser.add_argument("--ocr-max-attempts", type=int, default=6, help="Legacy base max OCR attempts.")
    parser.add_argument("--ocr-cooldown-frames", type=int, default=1, help="Legacy base OCR cooldown.")

    parser.add_argument("--ocr-gate-confidence", type=float, default=0.50, help="Minimum detection confidence for OCR gate.")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.45, help="Minimum OCR confidence for candidate acceptance.")
    parser.add_argument("--ocr-min-width", type=int, default=90, help="Minimum plate width for OCR gate.")
    parser.add_argument("--ocr-min-height", type=int, default=30, help="Minimum plate height for OCR gate.")
    parser.add_argument("--ocr-min-aspect-ratio", type=float, default=2.0, help="Minimum valid plate aspect ratio.")
    parser.add_argument("--ocr-max-aspect-ratio", type=float, default=6.0, help="Maximum valid plate aspect ratio.")
    parser.add_argument("--min-plate-area", type=int, default=2600, help="Minimum area for OCR gate.")
    parser.add_argument("--tiny-plate-area", type=int, default=280, help="Absolute lower bound area.")
    parser.add_argument("--small-plate-area", type=int, default=4200, help="Threshold for small-plate policy.")
    parser.add_argument("--medium-plate-area", type=int, default=11000, help="Threshold for medium-plate policy.")
    parser.add_argument("--small-plate-width", type=int, default=150, help="Width threshold for small-plate policy.")
    parser.add_argument("--medium-plate-width", type=int, default=240, help="Width threshold for medium-plate policy.")

    parser.add_argument("--small-plate-after-hits", type=int, default=2, help="Min age before OCR on small plates.")
    parser.add_argument("--medium-plate-after-hits", type=int, default=2, help="Min age before OCR on medium plates.")
    parser.add_argument("--large-plate-after-hits", type=int, default=2, help="Min age before OCR on large plates.")
    parser.add_argument("--small-plate-vote-threshold", type=int, default=2, help="Lock vote threshold for small plates.")
    parser.add_argument("--medium-plate-vote-threshold", type=int, default=2, help="Lock vote threshold for medium plates.")
    parser.add_argument("--large-plate-vote-threshold", type=int, default=2, help="Lock vote threshold for large plates.")
    parser.add_argument("--small-plate-max-attempts", type=int, default=12, help="Max OCR attempts for small plates.")
    parser.add_argument("--medium-plate-max-attempts", type=int, default=6, help="Max OCR attempts for medium plates.")
    parser.add_argument("--large-plate-max-attempts", type=int, default=6, help="Max OCR attempts for large plates.")
    parser.add_argument("--small-plate-cooldown-frames", type=int, default=2, help="Cooldown OCR retry for small plates.")
    parser.add_argument("--medium-plate-cooldown-frames", type=int, default=2, help="Cooldown OCR retry for medium plates.")
    parser.add_argument("--large-plate-cooldown-frames", type=int, default=1, help="Cooldown OCR retry for large plates.")
    parser.add_argument("--ocr-min-interval", type=int, default=2, help="Minimum frame interval between OCR attempts.")
    parser.add_argument("--max-plate-area-ratio", type=float, default=0.05, help="Reject very large detections as non-plates.")

    parser.add_argument("--ocr-history-size", type=int, default=10, help="OCR history buffer size per track.")
    parser.add_argument("--ocr-consensus-ratio", type=float, default=0.60, help="Minimum weighted consensus ratio.")
    parser.add_argument("--unreadable-after-age", type=int, default=30, help="Mark unresolved track unreadable after N frames.")

    parser.add_argument("--min-frame-stride", type=int, default=1, help="Minimum adaptive frame stride.")
    parser.add_argument("--max-frame-stride", type=int, default=3, help="Maximum adaptive frame stride.")
    parser.add_argument("--target-fps-low", type=float, default=20.0, help="Increase frame stride below this FPS.")
    parser.add_argument("--target-fps-high", type=float, default=30.0, help="Decrease frame stride above this FPS.")
    parser.add_argument("--stride-adjust-interval", type=int, default=20, help="Frames between adaptive stride updates.")
    parser.add_argument("--disable-adaptive-stride", action="store_true", help="Disable adaptive frame stride.")

    parser.add_argument("--track-ttl", type=int, default=90, help="Frames to keep inactive track metadata.")
    parser.add_argument("--bbox-history-size", type=int, default=5, help="History size for bbox smoothing.")

    parser.add_argument("--disable-clahe", action="store_true", help="Disable CLAHE before OCR.")
    parser.add_argument("--disable-sharpening", action="store_true", help="Disable sharpening OCR variant.")
    parser.add_argument("--disable-adaptive-threshold", action="store_true", help="Disable threshold OCR variant.")
    parser.add_argument("--disable-state-code-validation", action="store_true", help="Disable Indian state-code sanity filter.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display window.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    base_after_hits = max(1, args.ocr_after_hits)
    base_votes = max(1, args.ocr_vote_threshold)
    base_attempts = max(1, args.ocr_max_attempts)
    base_cooldown = max(1, args.ocr_cooldown_frames)

    return AppConfig(
        weights=args.weights,
        source=args.source,
        conf_threshold=min(max(args.conf, 0.01), 0.99),
        iou_threshold=min(max(args.iou, 0.05), 0.95),
        imgsz=max(320, args.imgsz),
        resize_width=max(0, args.resize_width),
        tracker=args.tracker,
        device=device,
        output_path=args.output,
        save_plates_dir=args.save_plates_dir,
        log_file=args.log_file,
        roi=args.roi,
        frame_stride=max(1, args.frame_stride),
        min_frame_stride=max(1, args.min_frame_stride),
        max_frame_stride=max(1, args.max_frame_stride),
        adaptive_stride=not args.disable_adaptive_stride,
        stride_adjust_interval=max(1, args.stride_adjust_interval),
        target_fps_low=max(1.0, args.target_fps_low),
        target_fps_high=max(1.0, args.target_fps_high),
        max_fps=max(0.0, args.max_fps),
        ocr_gate_confidence=min(max(args.ocr_gate_confidence, 0.0), 1.0),
        ocr_min_width=max(32, args.ocr_min_width),
        ocr_min_height=max(12, args.ocr_min_height),
        ocr_min_aspect_ratio=max(0.8, args.ocr_min_aspect_ratio),
        ocr_max_aspect_ratio=max(2.0, args.ocr_max_aspect_ratio),
        ocr_min_confidence=min(max(args.ocr_min_confidence, 0.0), 1.0),
        min_plate_area=max(64, args.min_plate_area),
        tiny_plate_area=max(64, args.tiny_plate_area),
        small_plate_area=max(128, args.small_plate_area),
        medium_plate_area=max(256, args.medium_plate_area),
        small_plate_width=max(64, args.small_plate_width),
        medium_plate_width=max(96, args.medium_plate_width),
        small_plate_after_hits=max(1, args.small_plate_after_hits),
        medium_plate_after_hits=max(1, args.medium_plate_after_hits),
        large_plate_after_hits=max(1, args.large_plate_after_hits),
        small_plate_vote_threshold=max(base_votes, args.small_plate_vote_threshold),
        medium_plate_vote_threshold=max(base_votes, args.medium_plate_vote_threshold),
        large_plate_vote_threshold=max(1, args.large_plate_vote_threshold),
        small_plate_max_attempts=max(base_attempts + 2, args.small_plate_max_attempts),
        medium_plate_max_attempts=max(base_attempts + 1, args.medium_plate_max_attempts),
        large_plate_max_attempts=max(1, args.large_plate_max_attempts),
        small_plate_cooldown_frames=max(1, args.small_plate_cooldown_frames),
        medium_plate_cooldown_frames=max(base_cooldown, args.medium_plate_cooldown_frames),
        large_plate_cooldown_frames=max(1, args.large_plate_cooldown_frames),
        ocr_min_interval=max(1, args.ocr_min_interval),
        max_plate_area_ratio=min(max(args.max_plate_area_ratio, 0.01), 0.60),
        track_ttl=max(1, args.track_ttl),
        bbox_history_size=max(1, args.bbox_history_size),
        ocr_history_size=max(3, args.ocr_history_size),
        ocr_consensus_ratio=min(max(args.ocr_consensus_ratio, 0.0), 1.0),
        unreadable_after_age=max(8, args.unreadable_after_age),
        use_clahe=not args.disable_clahe,
        use_sharpening=not args.disable_sharpening,
        adaptive_threshold=not args.disable_adaptive_threshold,
        enable_state_code_validation=not args.disable_state_code_validation,
        display=not args.no_display,
        use_half=device != "cpu",
    )


def resolve_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg

    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model(weights_path: Union[str, Path]) -> YOLO:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    try:
        model.fuse()
    except Exception:
        pass
    return model


def initialize_tracker(tracker_name: str = "bytetrack") -> str:
    tracker_name = tracker_name.lower()
    tracker_map = {
        "bytetrack": "bytetrack.yaml",
    }
    if tracker_name not in tracker_map:
        raise ValueError(f"Unsupported tracker '{tracker_name}'.")
    return tracker_map[tracker_name]


def initialize_ocr(use_gpu: bool) -> PaddleOCR:
    if _PADDLEOCR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PaddleOCR is not installed. Install it with: pip install paddleocr paddlepaddle"
        ) from _PADDLEOCR_IMPORT_ERROR

    ocr_device = "gpu:0" if use_gpu else "cpu"
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        device=ocr_device,
        enable_hpi=False,
        enable_mkldnn=False,
        enable_cinn=False,
    )


def classify_plate_size(plate_area: int, plate_width: int, config: AppConfig) -> str:
    if plate_area < config.small_plate_area or plate_width < config.small_plate_width:
        return "small"
    if plate_area < config.medium_plate_area or plate_width < config.medium_plate_width:
        return "medium"
    return "large"


def get_ocr_policy(plate_area: int, plate_width: int, config: AppConfig) -> OCRPolicy:
    plate_size = classify_plate_size(plate_area, plate_width, config)
    if plate_size == "small":
        return OCRPolicy(
            plate_size=plate_size,
            min_area=max(config.min_plate_area, config.tiny_plate_area),
            min_width=max(60, int(config.ocr_min_width * 0.75)),
            min_height=config.ocr_min_height,
            min_aspect_ratio=config.ocr_min_aspect_ratio,
            max_aspect_ratio=config.ocr_max_aspect_ratio,
            scale_factor=max(3.0, config.small_plate_scale),
            ocr_after_frames=config.small_plate_after_hits,
            vote_threshold=config.small_plate_vote_threshold,
            max_attempts=config.small_plate_max_attempts,
            min_confidence=config.ocr_min_confidence,
            min_detection_confidence=config.ocr_gate_confidence,
            sharpness_threshold=config.small_plate_blur_threshold,
            cooldown_frames=config.small_plate_cooldown_frames,
            lock_consensus=max(config.ocr_consensus_ratio, config.lock_consensus_small),
        )
    if plate_size == "medium":
        return OCRPolicy(
            plate_size=plate_size,
            min_area=max(config.min_plate_area, config.tiny_plate_area),
            min_width=config.ocr_min_width,
            min_height=config.ocr_min_height,
            min_aspect_ratio=config.ocr_min_aspect_ratio,
            max_aspect_ratio=config.ocr_max_aspect_ratio,
            scale_factor=config.medium_plate_scale,
            ocr_after_frames=config.medium_plate_after_hits,
            vote_threshold=config.medium_plate_vote_threshold,
            max_attempts=config.medium_plate_max_attempts,
            min_confidence=config.ocr_min_confidence,
            min_detection_confidence=config.ocr_gate_confidence,
            sharpness_threshold=config.medium_plate_blur_threshold,
            cooldown_frames=config.medium_plate_cooldown_frames,
            lock_consensus=max(config.ocr_consensus_ratio, config.lock_consensus_medium),
        )

    return OCRPolicy(
        plate_size=plate_size,
        min_area=config.min_plate_area,
        min_width=config.ocr_min_width,
        min_height=config.ocr_min_height,
        min_aspect_ratio=config.ocr_min_aspect_ratio,
        max_aspect_ratio=config.ocr_max_aspect_ratio,
        scale_factor=config.large_plate_scale,
        ocr_after_frames=config.large_plate_after_hits,
        vote_threshold=config.large_plate_vote_threshold,
        max_attempts=config.large_plate_max_attempts,
        min_confidence=config.ocr_min_confidence,
        min_detection_confidence=config.ocr_gate_confidence,
        sharpness_threshold=config.blur_threshold,
        cooldown_frames=config.large_plate_cooldown_frames,
        lock_consensus=max(config.ocr_consensus_ratio, config.lock_consensus_large),
    )


def build_enhanced_gray(crop: np.ndarray, scale_factor: float, use_clahe: bool) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if scale_factor > 1.0:
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def compute_blur_score(crop: np.ndarray, scale_factor: float, use_clahe: bool) -> float:
    sharpness_scale = max(1.0, min(scale_factor, 2.0))
    gray = build_enhanced_gray(crop, scale_factor=sharpness_scale, use_clahe=use_clahe)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess_plate_crop(crop: np.ndarray, policy: OCRPolicy, config: AppConfig) -> List[np.ndarray]:
    gray = build_enhanced_gray(crop, scale_factor=policy.scale_factor, use_clahe=config.use_clahe)
    sharpened = gray
    if config.use_sharpening:
        sharpen_kernel = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        sharpened = cv2.filter2D(gray, ddepth=-1, kernel=sharpen_kernel)

    variants = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]
    if config.use_sharpening:
        variants.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))
    if config.adaptive_threshold:
        thresholded = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            6,
        )
        variants.append(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR))
    return variants


def sanitize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def normalize_digit_token(token: str) -> str:
    replacements = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }
    converted = "".join(replacements.get(ch, ch) for ch in token)
    return converted if converted.isdigit() else ""


def normalize_alpha_token(token: str) -> str:
    replacements = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }
    converted = "".join(replacements.get(ch, ch) for ch in token)
    return converted if converted.isalpha() else ""


def has_valid_state_code(candidate: str) -> bool:
    if len(candidate) < 2:
        return False
    if candidate[2:4] == "BH":
        return True
    return candidate[:2] in STATE_CODES


def extract_valid_plate_text(text: str) -> str:
    cleaned = sanitize_plate_text(text)
    if len(cleaned) < 8:
        return ""
    best_candidate = ""
    for start in range(len(cleaned)):
        for end in range(start + 8, min(len(cleaned), start + 11) + 1):
            chunk = cleaned[start:end]
            for district_len in (2, 1):
                state = normalize_alpha_token(chunk[:2])
                district = normalize_digit_token(chunk[2 : 2 + district_len])
                number = normalize_digit_token(chunk[-4:])
                series = normalize_alpha_token(chunk[2 + district_len : -4])
                if not state or not district or not series or not number:
                    continue
                candidate = f"{state}{district}{series}{number}"
                if INDIAN_PLATE_REGEX.match(candidate) and has_valid_state_code(candidate):
                    return candidate
                if INDIAN_PLATE_REGEX.match(candidate):
                    best_candidate = candidate

    for start in range(len(cleaned)):
        for end in range(start + 9, min(len(cleaned), start + 10) + 1):
            chunk = cleaned[start:end]
            year = normalize_digit_token(chunk[:2])
            bh = normalize_alpha_token(chunk[2:4])
            serial = normalize_digit_token(chunk[4:8])
            suffix = normalize_alpha_token(chunk[8:])
            if not year or bh != "BH" or not serial or not suffix:
                continue
            candidate = f"{year}BH{serial}{suffix}"
            if BH_PLATE_REGEX.match(candidate):
                return candidate
    return best_candidate


def extract_plate_crop(frame: np.ndarray, bbox: BBox, padding_ratio: float = 0.08) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    h, w = frame.shape[:2]

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def extract_ocr_text_scores(ocr_result: Any) -> List[Tuple[str, float]]:
    extracted: List[Tuple[str, float]] = []
    if not ocr_result:
        return extracted

    first_item = ocr_result[0]
    if hasattr(first_item, "get"):
        rec_texts = first_item.get("rec_texts", []) or []
        rec_scores = first_item.get("rec_scores", []) or []
        for text, score in zip(rec_texts, rec_scores):
            try:
                extracted.append((str(text), float(score)))
            except (TypeError, ValueError):
                extracted.append((str(text), 0.0))
        return extracted

    lines = ocr_result[0] if isinstance(ocr_result[0], list) else ocr_result
    for line in lines:
        if not line or len(line) < 2:
            continue

        text_info = line[1]
        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
            continue

        try:
            extracted.append((str(text_info[0]), float(text_info[1])))
        except (TypeError, ValueError):
            extracted.append((str(text_info[0]), 0.0))
    return extracted


def run_ocr_on_plate(crop: np.ndarray, ocr_engine: PaddleOCR, config: AppConfig, policy: OCRPolicy) -> Tuple[str, float, bool]:
    best_text = ""
    best_confidence = 0.0
    weak_candidate = False

    for variant in preprocess_plate_crop(crop, policy=policy, config=config):
        ocr_result = ocr_engine.predict(variant)
        if not ocr_result:
            continue

        fragments: List[str] = []
        scores: List[float] = []

        for raw_text, raw_score in extract_ocr_text_scores(ocr_result):
            cleaned_fragment = sanitize_plate_text(raw_text)
            if len(cleaned_fragment) < 3:
                continue

            fragments.append(cleaned_fragment)
            scores.append(raw_score)

        if not fragments:
            continue

        candidate_text = extract_valid_plate_text("".join(fragments))
        if not candidate_text:
            for fragment in fragments:
                candidate_text = extract_valid_plate_text(fragment)
                if candidate_text:
                    break

        if not candidate_text:
            continue
        if config.enable_state_code_validation and not has_valid_state_code(candidate_text):
            continue

        confidence = float(np.mean(scores)) if scores else 0.0
        if confidence < 0.30:
            continue

        if confidence > best_confidence:
            best_text = candidate_text
            best_confidence = confidence
            weak_candidate = confidence < 0.50

    return best_text, best_confidence, weak_candidate


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame

    scale = target_width / float(frame.shape[1])
    target_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def clip_bbox(bbox: np.ndarray, frame_shape: Tuple[int, int, int]) -> BBox:
    x1, y1, x2, y2 = bbox.astype(int)
    height, width = frame_shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def can_use_paddle_gpu() -> bool:
    try:
        import paddle

        return bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False


def parse_video_source(source: str) -> Union[int, str]:
    return int(source) if source.isdigit() else source


def parse_roi_bbox(frame_shape: Tuple[int, int, int], roi_spec: Optional[str]) -> Optional[BBox]:
    if not roi_spec:
        return None

    values = [value.strip() for value in roi_spec.split(",")]
    if len(values) != 4:
        raise ValueError("ROI must be provided as x1,y1,x2,y2.")

    numbers = [float(value) for value in values]
    height, width = frame_shape[:2]
    if all(0.0 <= value <= 1.0 for value in numbers):
        x1, y1, x2, y2 = (
            int(round(numbers[0] * width)),
            int(round(numbers[1] * height)),
            int(round(numbers[2] * width)),
            int(round(numbers[3] * height)),
        )
    else:
        x1, y1, x2, y2 = (int(round(value)) for value in numbers)

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI coordinates must define a non-empty rectangle.")
    return x1, y1, x2, y2


def get_detection_frame(working_frame: np.ndarray, config: AppConfig) -> Tuple[np.ndarray, Tuple[int, int], Optional[BBox]]:
    roi_bbox = parse_roi_bbox(working_frame.shape, config.roi)
    if roi_bbox is None:
        return working_frame, (0, 0), None

    x1, y1, x2, y2 = roi_bbox
    return working_frame[y1:y2, x1:x2], (x1, y1), roi_bbox


def offset_bbox(bbox: BBox, offset: Tuple[int, int], frame_shape: Tuple[int, int, int]) -> BBox:
    ox, oy = offset
    x1, y1, x2, y2 = bbox
    shifted = np.array([x1 + ox, y1 + oy, x2 + ox, y2 + oy], dtype=np.int32)
    return clip_bbox(shifted, frame_shape)


def log_plate_detection(logger: Optional[logging.Logger], track_id: int, plate_text: str, confidence: float) -> None:
    if logger and plate_text:
        logger.info("track_id=%s plate=%s ocr_confidence=%.3f", track_id, plate_text, confidence)


def maybe_save_plate_crop(
    crop: np.ndarray,
    config: AppConfig,
    track_id: int,
    plate_text: str,
    frame_index: int,
) -> None:
    if config.save_plates_dir is None or crop.size == 0:
        return

    config.save_plates_dir.mkdir(parents=True, exist_ok=True)
    safe_text = plate_text or "unreadable"
    file_name = f"track_{track_id:04d}_{safe_text}_{frame_index:06d}.jpg"
    cv2.imwrite(str(config.save_plates_dir / file_name), crop)


def cleanup_stale_tracks(runtime: RuntimeState, frame_index: int, ttl: int) -> None:
    expired_ids = [
        track_id
        for track_id, state in runtime.track_states.items()
        if frame_index - state.last_seen_frame > ttl
    ]
    for track_id in expired_ids:
        runtime.track_states.pop(track_id, None)
        runtime.plate_cache.pop(track_id, None)


def update_bbox_history(state: TrackState, bbox: BBox, history_size: int) -> BBox:
    state.last_bbox = bbox
    state.bbox_history.append(bbox)
    while len(state.bbox_history) > history_size:
        state.bbox_history.popleft()

    bbox_array = np.asarray(state.bbox_history, dtype=np.float32)
    smoothed_bbox = tuple(np.rint(bbox_array.mean(axis=0)).astype(np.int32).tolist())
    state.smoothed_bbox = smoothed_bbox
    return smoothed_bbox


def update_ocr_history(state: TrackState, frame_index: int, plate_text: str, confidence: float, history_size: int) -> None:
    state.ocr_history.append((frame_index, plate_text, confidence))
    while len(state.ocr_history) > history_size:
        _, old_text, old_confidence = state.ocr_history.popleft()
        if old_text in state.ocr_votes:
            state.ocr_votes[old_text] -= 1
            if state.ocr_votes[old_text] <= 0:
                state.ocr_votes.pop(old_text, None)
        if old_text in state.ocr_score_totals:
            state.ocr_score_totals[old_text] -= old_confidence
            if state.ocr_score_totals[old_text] <= 1e-6:
                state.ocr_score_totals.pop(old_text, None)


def _normalized_for_similarity(text: str) -> str:
    confusable_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }
    return "".join(confusable_map.get(ch, ch) for ch in text)


def plate_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def merge_vote_key(state: TrackState, plate_text: str, threshold: float = 0.55) -> str:
    best_key = plate_text
    best_similarity = 0.0
    for existing_key in state.ocr_votes.keys():
        similarity = plate_similarity(
            _normalized_for_similarity(existing_key),
            _normalized_for_similarity(plate_text),
        )
        if similarity >= threshold and similarity > best_similarity:
            best_key = existing_key
            best_similarity = similarity
    return best_key


def add_ocr_vote(
    state: TrackState,
    frame_index: int,
    plate_text: str,
    confidence: float,
    history_size: int,
) -> None:
    vote_key = merge_vote_key(state, plate_text, threshold=0.70)
    score_boost = 0.05 if vote_key != plate_text else 0.0
    merged_score = confidence + score_boost + (state.hits * 0.01)
    if confidence < 0.5:
        merged_score -= 0.05
    merged_score = max(0.0, min(1.0, merged_score))
    state.ocr_votes[vote_key] += 1
    state.ocr_score_totals[vote_key] = state.ocr_score_totals.get(vote_key, 0.0) + merged_score
    update_ocr_history(state, frame_index, vote_key, merged_score, history_size)


def get_best_vote(state: TrackState) -> Tuple[str, int, float, float]:
    if not state.ocr_votes:
        return "", 0, 0.0, 0.0

    best_text = max(
        state.ocr_votes.keys(),
        key=lambda text: (
            state.ocr_score_totals.get(text, 0.0),
            state.ocr_score_totals.get(text, 0.0) / max(1, state.ocr_votes[text]),
            state.ocr_votes[text],
        ),
    )
    votes = state.ocr_votes[best_text]
    average_confidence = state.ocr_score_totals.get(best_text, 0.0) / max(1, votes)
    total_weight = float(sum(state.ocr_score_totals.values()))
    consensus_ratio = (state.ocr_score_totals.get(best_text, 0.0) / total_weight) if total_weight > 0 else 0.0
    return best_text, votes, average_confidence, consensus_ratio


def get_track_age(state: TrackState, frame_index: int) -> int:
    return frame_index - state.first_seen_frame + 1


def evaluate_ocr_gate(
    state: TrackState,
    crop: Optional[np.ndarray],
    bbox: BBox,
    track_id: int,
    detection_confidence: float,
    policy: OCRPolicy,
    frame_index: int,
    config: AppConfig,
    frame_shape: Tuple[int, int, int],
    force_ocr: bool = False,
) -> Tuple[bool, str, float]:
    track_age = get_track_age(state, frame_index)
    edge_density = -1.0
    x1, y1, x2, y2 = bbox
    plate_width = max(0, x2 - x1)
    plate_height = max(0, y2 - y1)
    aspect_ratio = plate_width / max(1, plate_height)
    plate_area = plate_width * plate_height
    soft_flags: List[str] = []

    def _decision(allowed: bool, reason: str, blur_score: float) -> Tuple[bool, str, float]:
        print(
            f"[OCR DECISION] ID={track_id}, age={track_age}, "
            f"force={force_ocr}, edge={edge_density:.4f}, decision={reason}"
        )
        return allowed, reason, blur_score

    if state.status == READING_STATUS:
        if (
            1.5 <= aspect_ratio <= 7.5
            and plate_width > 60
            and plate_area > config.tiny_plate_area
            and detection_confidence >= 0.35
        ):
            soft_flags.append("reading_override")

    if state.status in {LOCKED_STATUS, WEAK_LOCKED_STATUS} or state.ocr_locked:
        return _decision(False, "locked", state.last_blur_score)
    if crop is None:
        return _decision(False, "no_crop", 0.0)

    # Always allow early OCR attempts so tracks can bootstrap OCR history quickly.
    if track_age <= 2:
        return _decision(True, "early_attempt", state.last_blur_score)

    improving = False
    if len(state.ocr_history) >= 2:
        recent = [entry[2] for entry in list(state.ocr_history)[-3:]]
        improving = recent[-1] >= (sum(recent[:-1]) / max(1, len(recent) - 1)) - 0.03
    max_attempts_with_grace = policy.max_attempts + 3
    if state.ocr_attempts >= policy.max_attempts and not (improving and state.ocr_attempts < max_attempts_with_grace):
        if not force_ocr:
            return _decision(False, "max_attempts", state.last_blur_score)
    if policy.plate_size == "small":
        required_interval = 1
    elif policy.plate_size == "medium":
        required_interval = 2
    else:
        required_interval = config.ocr_min_interval
    if (
        frame_index - state.last_ocr_frame < required_interval
        and not force_ocr
        and state.status != READING_STATUS
    ):
        return _decision(False, "cooldown", state.last_blur_score)
    if track_age < policy.ocr_after_frames:
        soft_flags.append("waiting_age")
    if force_ocr:
        soft_flags.append("forced_fallback")

    frame_area = max(1, frame_shape[0] * frame_shape[1])
    if plate_area <= 25 or plate_width <= 5 or plate_height <= 5:
        return _decision(False, "tiny_area", state.last_blur_score)
    if plate_area > int(frame_area * config.max_plate_area_ratio):
        soft_flags.append("too_large")

    if plate_width < policy.min_width:
        if plate_width >= 60 or force_ocr:
            soft_flags.append("soft_small")
        else:
            soft_flags.append("hard_small")
    if detection_confidence < policy.min_detection_confidence:
        if detection_confidence >= 0.40 or force_ocr:
            soft_flags.append("soft_conf")
        else:
            soft_flags.append("hard_conf")
    if plate_height < policy.min_height:
        soft_flags.append("small_height")
    if plate_area < policy.min_area:
        if plate_area >= int(policy.min_area * 0.60) or force_ocr:
            soft_flags.append("soft_area")
        else:
            soft_flags.append("small_area")
    if aspect_ratio < 1.5:
        return _decision(False, "non_plate_shape", state.last_blur_score)
    if aspect_ratio > 7.5:
        soft_flags.append("bad_aspect")

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray_crop)) < 3.0:
        return _decision(False, "blank_crop", state.last_blur_score)
    edge_density = float(np.mean(cv2.Canny(gray_crop, 100, 200)) / 255.0)
    if edge_density < 0.01:
        soft_flags.append("low_text_texture")

    sharpness_score = compute_blur_score(crop, scale_factor=policy.scale_factor, use_clahe=config.use_clahe)
    if sharpness_score < policy.sharpness_threshold:
        soft_flags.append("low_quality_blur")
    gate_reason = "|".join(soft_flags) if soft_flags else "ready"

    hard_reject = (
        "tiny_area" in gate_reason or
        "blank_crop" in gate_reason
    )

    if hard_reject:
        return _decision(False, gate_reason, sharpness_score)

    return _decision(True, gate_reason, sharpness_score)


def finalize_track_text(
    track_id: int,
    state: TrackState,
    runtime: RuntimeState,
    logger: Optional[logging.Logger],
    crop: Optional[np.ndarray],
    config: AppConfig,
    policy: OCRPolicy,
    frame_index: int,
) -> None:
    best_text, votes, average_confidence, consensus_ratio = get_best_vote(state)
    state.last_vote_count = votes
    exhausted = state.ocr_attempts >= policy.max_attempts
    track_age = get_track_age(state, frame_index)
    timed_out = track_age >= config.unreadable_after_age
    strong_lock = votes >= 2 and average_confidence >= 0.45
    weak_lock = votes >= 2 and average_confidence >= 0.30

    if not strong_lock and not weak_lock and not exhausted and not timed_out:
        state.status = READING_STATUS
        return

    state.ocr_locked = strong_lock or weak_lock
    if best_text and (strong_lock or weak_lock):
        state.plate_text = best_text
        state.ocr_confidence = average_confidence
        state.consensus_ratio = consensus_ratio
        state.status = LOCKED_STATUS if strong_lock else WEAK_LOCKED_STATUS
        state.last_gate_reason = "stable"
        runtime.plate_cache[track_id] = best_text
        if not state.logged:
            runtime.ocr_success_total += 1
            log_plate_detection(logger, track_id, best_text, average_confidence)
            state.logged = True

        if crop is not None and not state.crop_saved:
            maybe_save_plate_crop(crop, config, track_id, best_text, frame_index)
            state.crop_saved = True
    elif (exhausted or timed_out) and best_text and votes >= 2 and average_confidence >= 0.35:
        state.status = WEAK_LOCKED_STATUS
        state.plate_text = best_text
        state.ocr_confidence = average_confidence
        state.consensus_ratio = consensus_ratio
        state.last_gate_reason = "best_guess"
        runtime.plate_cache[track_id] = best_text
    elif exhausted or timed_out:
        state.status = UNREADABLE_STATUS
        state.plate_text = UNREADABLE_TEXT
        state.ocr_confidence = 0.0
        state.consensus_ratio = consensus_ratio
        state.last_gate_reason = "insufficient_quality"
        runtime.plate_cache[track_id] = UNREADABLE_TEXT


def build_visible_detections(runtime: RuntimeState, frame_index: int, max_age_frames: int) -> List[PlateDetection]:
    detections: List[PlateDetection] = []
    for track_id, state in runtime.track_states.items():
        if state.smoothed_bbox is None or frame_index - state.last_seen_frame > max_age_frames:
            continue

        display_text = state.plate_text if state.status in {LOCKED_STATUS, WEAK_LOCKED_STATUS} else ""
        if state.status == UNREADABLE_STATUS:
            display_text = UNREADABLE_TEXT
        detections.append(
            PlateDetection(
                bbox=state.smoothed_bbox,
                track_id=track_id,
                confidence=state.last_confidence,
                plate_text=display_text,
                status=state.status,
                reused=frame_index > state.last_seen_frame,
                plate_size=state.plate_size,
                blur_score=state.last_blur_score,
                ocr_confidence=state.ocr_confidence,
                consensus_ratio=state.consensus_ratio,
                gate_reason=state.last_gate_reason,
                vote_count=state.last_vote_count,
            )
        )

    detections.sort(key=lambda detection: detection.track_id)
    return detections


def adjust_frame_stride(runtime: RuntimeState, config: AppConfig, frame_index: int) -> None:
    if not config.adaptive_stride or len(runtime.fps_history) < 10:
        return
    if frame_index - runtime.last_stride_adjust_frame < config.stride_adjust_interval:
        return

    avg_fps = float(np.mean(runtime.fps_history))
    target_high = config.target_fps_high
    if config.max_fps > 0:
        target_high = min(target_high, max(config.target_fps_low + 1.0, config.max_fps * 0.95))

    if avg_fps < config.target_fps_low and runtime.current_frame_stride < config.max_frame_stride:
        runtime.current_frame_stride += 1
        runtime.last_stride_adjust_frame = frame_index
    elif avg_fps > target_high and runtime.current_frame_stride > config.min_frame_stride:
        runtime.current_frame_stride -= 1
        runtime.last_stride_adjust_frame = frame_index


def should_process_detection(frame_index: int, frame_stride: int) -> bool:
    return frame_stride <= 1 or (frame_index - 1) % frame_stride == 0


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    tracker_config: str,
    ocr_engine: PaddleOCR,
    runtime: RuntimeState,
    config: AppConfig,
    frame_index: int,
    logger: Optional[logging.Logger],
    run_detection: bool,
) -> Tuple[np.ndarray, List[PlateDetection]]:
    working_frame = resize_frame(frame, config.resize_width)
    cleanup_stale_tracks(runtime, frame_index, config.track_ttl)
    detection_frame, detection_offset, roi_bbox = get_detection_frame(working_frame, config)
    runtime.last_roi_bbox = roi_bbox

    if not run_detection:
        return working_frame, build_visible_detections(runtime, frame_index, max(0, runtime.current_frame_stride - 1))

    runtime.processed_frames += 1
    if detection_frame.size == 0:
        return working_frame, build_visible_detections(runtime, frame_index, 1)

    results = model.track(
        detection_frame,
        persist=True,
        tracker=tracker_config,
        conf=config.conf_threshold,
        iou=config.iou_threshold,
        imgsz=config.imgsz,
        device=config.device,
        half=config.use_half,
        verbose=False,
    )

    if not results:
        return working_frame, build_visible_detections(runtime, frame_index, max(1, runtime.current_frame_stride))

    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return working_frame, build_visible_detections(runtime, frame_index, max(1, runtime.current_frame_stride))

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    track_ids = (
        boxes.id.detach().cpu().numpy().astype(int).tolist()
        if boxes.id is not None
        else list(range(len(xyxy)))
    )

    for bbox_array, confidence, track_id in zip(xyxy, confidences, track_ids):
        bbox = clip_bbox(bbox_array, detection_frame.shape)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        bbox = offset_bbox(bbox, detection_offset, working_frame.shape)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        state = runtime.track_states.setdefault(track_id, TrackState())
        if state.hits == 0:
            state.first_seen_frame = frame_index
        state.hits += 1
        state.last_seen_frame = frame_index
        state.last_confidence = float(confidence)
        update_bbox_history(state, bbox, config.bbox_history_size)

        crop = extract_plate_crop(working_frame, bbox)
        plate_width = max(0, x2 - x1)
        plate_area = plate_width * max(0, y2 - y1)
        force_due_size_boost = False
        if plate_area > 0:
            previous_max_area = state.max_plate_area_seen
            state.max_plate_area_seen = max(state.max_plate_area_seen, plate_area)
            if previous_max_area > 0 and plate_area >= int(previous_max_area * 1.30):
                state.ocr_attempts = 0
                state.burst_count = 0
                state.last_gate_reason = "size_boost_reset"
                state.last_ocr_frame = -100
                force_due_size_boost = True
        policy = get_ocr_policy(plate_area, plate_width, config)
        state.plate_size = policy.plate_size
        track_age = get_track_age(state, frame_index)
        state.track_age = track_age
        state.burst_count = getattr(state, "burst_count", 0)
        force_due_size_boost = force_due_size_boost  # keep existing logic
        force_due_periodic = (
            state.status == READING_STATUS
            and config.force_ocr_every_n_frames > 0
            and track_age > 0
            and track_age % config.force_ocr_every_n_frames == 0
        )
        force_ocr = force_due_size_boost or force_due_periodic

        # Initial burst only (important for bootstrapping OCR)
        if state.burst_count < 3:
            force_ocr = True
            state.burst_count += 1
        sharpness_score = state.last_blur_score
        if crop is not None and crop.size > 0 and (state.last_blur_score > 0.0 or state.status == READING_STATUS):
            sharpness_score = compute_blur_score(
                crop,
                scale_factor=policy.scale_factor,
                use_clahe=config.use_clahe,
            )

        can_run_ocr, gate_reason, blur_score = evaluate_ocr_gate(
            state=state,
            crop=crop,
            bbox=bbox,
            track_id=track_id,
            detection_confidence=float(confidence),
            policy=policy,
            frame_index=frame_index,
            config=config,
            frame_shape=working_frame.shape,
            force_ocr=force_ocr,
        )
        state.last_gate_reason = gate_reason
        state.last_blur_score = max(blur_score, sharpness_score)

        if can_run_ocr:
            state.last_ocr_frame = frame_index
            state.ocr_attempts += 1
            runtime.ocr_attempts_total += 1

            plate_text, ocr_confidence, weak_candidate = run_ocr_on_plate(crop, ocr_engine, config, policy)
            if state.last_blur_score < policy.sharpness_threshold:
                ocr_confidence *= 0.85
            if plate_text and ocr_confidence >= 0.30:
                add_ocr_vote(
                    state=state,
                    frame_index=frame_index,
                    plate_text=plate_text,
                    confidence=ocr_confidence,
                    history_size=config.ocr_history_size,
                )
                if weak_candidate:
                    state.last_gate_reason = f"{state.last_gate_reason}|weak_candidate" if state.last_gate_reason else "weak_candidate"
        else:
            runtime.ocr_gated_total += 1
            if gate_reason.startswith("hard_") or gate_reason in {"small_height", "small_area", "bad_aspect", "max_attempts"}:
                runtime.ocr_strict_skip_total += 1

        finalize_track_text(
            track_id=track_id,
            state=state,
            runtime=runtime,
            logger=logger,
            crop=crop,
            config=config,
            policy=policy,
            frame_index=frame_index,
        )

    return working_frame, build_visible_detections(runtime, frame_index, 0)


def draw_metric_line(frame: np.ndarray, text: str, line_index: int, color: Tuple[int, int, int]) -> None:
    x = 12
    y = 28 + line_index * 26
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(
        frame,
        (x - 6, y - text_size[1] - 8),
        (x + text_size[0] + 8, y + 8),
        (20, 20, 20),
        -1,
    )
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_results(
    frame: np.ndarray,
    detections: List[PlateDetection],
    fps: float,
    runtime: RuntimeState,
    run_detection: bool,
) -> np.ndarray:
    canvas = frame.copy()
    if runtime.last_roi_bbox is not None:
        rx1, ry1, rx2, ry2 = runtime.last_roi_bbox
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (80, 80, 220), 1)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        has_text = bool(detection.plate_text) and detection.plate_text != UNREADABLE_TEXT
        if detection.status == LOCKED_STATUS:
            color = (60, 220, 140)
        elif detection.status == WEAK_LOCKED_STATUS:
            color = (90, 200, 255)
        elif detection.status == UNREADABLE_STATUS:
            color = (80, 90, 255)
        else:
            color = (0, 200, 255)
        if detection.reused and detection.status == READING_STATUS:
            color = (160, 210, 255)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"ID {detection.track_id} | det {detection.confidence:.2f}"
        if detection.status in {LOCKED_STATUS, WEAK_LOCKED_STATUS} and has_text:
            lock_tag = "locked" if detection.status == LOCKED_STATUS else "weak_locked"
            label = (
                f"{label} | {detection.plate_text} | {lock_tag} "
                f"| ocr {detection.ocr_confidence:.2f} | v {detection.vote_count}"
            )
        elif detection.status == UNREADABLE_STATUS:
            label = f"{label} | unreadable"
        else:
            gate_reason = detection.gate_reason if detection.gate_reason else "reading"
            label = f"{label} | reading | {gate_reason} | v {detection.vote_count}"
        label = f"{label} | blur {detection.blur_score:.0f}"
        if detection.plate_size != "unknown":
            label = f"{label} | {detection.plate_size}"

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_width, text_height = text_size
        text_top = max(0, y1 - text_height - 10)
        cv2.rectangle(
            canvas,
            (x1, text_top),
            (x1 + text_width + 8, text_top + text_height + 8),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, text_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    ocr_success_rate = ((runtime.ocr_success_total / runtime.ocr_attempts_total) * 100.0) if runtime.ocr_attempts_total else 0.0
    total_gate_checks = runtime.ocr_attempts_total + runtime.ocr_gated_total
    gate_skip_rate = ((runtime.ocr_gated_total / total_gate_checks) * 100.0) if total_gate_checks else 0.0
    mode_label = "DET" if run_detection else "REUSE"
    draw_metric_line(canvas, f"FPS: {fps:.1f}", 0, (50, 240, 50))
    draw_metric_line(canvas, f"Tracks: {len(detections)}", 1, (255, 220, 120))
    draw_metric_line(canvas, f"OCR lock: {runtime.ocr_success_total}/{runtime.ocr_attempts_total} ({ocr_success_rate:.0f}%)", 2, (255, 200, 120))
    draw_metric_line(canvas, f"Gate skip: {runtime.ocr_gated_total}/{total_gate_checks} ({gate_skip_rate:.0f}%)", 3, (120, 220, 255))
    draw_metric_line(canvas, f"Strict skip: {runtime.ocr_strict_skip_total}", 4, (220, 200, 120))
    draw_metric_line(canvas, f"Mode: {mode_label} | Stride: {runtime.current_frame_stride}", 5, (220, 220, 120))
    if runtime.last_roi_bbox is not None:
        draw_metric_line(canvas, "ROI: ON", 6, (220, 160, 255))
    return canvas


def create_video_writer(output_path: Path, frame_shape: Tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    fps = fps if fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def setup_logger(log_file: Optional[Path]) -> Optional[logging.Logger]:
    if log_file is None:
        return None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("anpr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def main() -> None:
    config = parse_args()
    logger = setup_logger(config.log_file)

    cv2.setUseOptimized(True)

    model = load_model(config.weights)
    tracker_config = initialize_tracker(config.tracker)
    try:
        ocr_engine = initialize_ocr(use_gpu=config.device != "cpu" and can_use_paddle_gpu())
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize PaddleOCR. Ensure paddleocr/paddlepaddle are installed and OCR models are available."
        ) from exc

    runtime = RuntimeState()
    config.max_frame_stride = max(config.min_frame_stride, config.max_frame_stride)
    config.frame_stride = min(max(config.frame_stride, config.min_frame_stride), config.max_frame_stride)
    config.small_plate_area = max(config.tiny_plate_area + 1, config.small_plate_area)
    config.medium_plate_area = max(config.small_plate_area + 1, config.medium_plate_area)
    config.small_plate_width = max(config.ocr_min_width + 1, config.small_plate_width)
    config.medium_plate_width = max(config.small_plate_width + 1, config.medium_plate_width)
    if config.ocr_max_aspect_ratio <= config.ocr_min_aspect_ratio:
        config.ocr_max_aspect_ratio = config.ocr_min_aspect_ratio + 1.0
    if config.target_fps_high <= config.target_fps_low:
        config.target_fps_high = config.target_fps_low + 4.0
    runtime.current_frame_stride = config.frame_stride
    source = parse_video_source(config.source)
    if isinstance(source, str) and not Path(source).exists():
        raise FileNotFoundError(f"Video source not found: {source}")

    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {config.source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    writer: Optional[cv2.VideoWriter] = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            runtime.total_frames += 1
            loop_start = time.perf_counter()
            adjust_frame_stride(runtime, config, frame_index)
            run_detection = should_process_detection(frame_index, runtime.current_frame_stride)

            processed_frame, detections = process_frame(
                frame=frame,
                model=model,
                tracker_config=tracker_config,
                ocr_engine=ocr_engine,
                runtime=runtime,
                config=config,
                frame_index=frame_index,
                logger=logger,
                run_detection=run_detection,
            )

            if config.max_fps > 0:
                min_frame_time = 1.0 / config.max_fps
                processing_elapsed = time.perf_counter() - loop_start
                if processing_elapsed < min_frame_time:
                    time.sleep(min_frame_time - processing_elapsed)

            elapsed = max(time.perf_counter() - loop_start, 1e-6)
            runtime.fps_history.append(1.0 / elapsed)
            avg_fps = float(np.mean(runtime.fps_history)) if runtime.fps_history else 0.0

            annotated_frame = draw_results(
                processed_frame,
                detections,
                avg_fps,
                runtime,
                run_detection=run_detection,
            )

            if writer is None and config.output_path is not None:
                writer = create_video_writer(config.output_path, annotated_frame.shape, source_fps)
            if writer is not None:
                writer.write(annotated_frame)

            if config.display:
                cv2.imshow("ANPR - YOLOv8 + ByteTrack + PaddleOCR", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if config.display:
            cv2.destroyAllWindows()
        if logger:
            logger.info(
                "summary ocr_attempts=%s ocr_locked=%s ocr_gated=%s strict_gated=%s",
                runtime.ocr_attempts_total,
                runtime.ocr_success_total,
                runtime.ocr_gated_total,
                runtime.ocr_strict_skip_total,
            )


if __name__ == "__main__":
    main()
