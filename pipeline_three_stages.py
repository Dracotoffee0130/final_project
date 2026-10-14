from collections import defaultdict, deque
import csv
from pathlib import Path
import re
import cv2
import easyocr
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# =====================================================================
# 1. PATHS & CONFIGURATION 
# =====================================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\helmet_project")

STAGE1_WEIGHTS = str(BASE_DIR / "weights" / "yolo26s_motorcycle_detector.pt")
STAGE2_WEIGHTS = str(BASE_DIR / "weights" / "best.pt")
STAGE3_PLATE_WEIGHTS = str(BASE_DIR / "weights" / "plate_detection.pt")

INPUT_MEDIA_DIR = str(BASE_DIR / "osf_images")
OUTPUT_VIDEOS_DIR = str(BASE_DIR / "output_videos")

# Leave empty [] to automatically process ALL mobile videos (.mp4/.mov)

TARGET_CLIPS = []

# Stage 1 & Stage 2 Hyperparameters
DEFAULT_OSF_FPS = 10.0     # Used ONLY for OSF image sequences without FPS metadata
STAGE1_CONF = 0.35         # Motorcycle detection confidence
STAGE1_IOU = 0.50          # Motorcycle NMS IoU
BOX_DILATION = 0.18        # +18% expansion to preserve rider heads/helmets
STAGE2_IMGSZ = 256         # Stage 2 classifier input resolution
VOTING_DURATION_SEC = 0.4  # 0.4s weighted voting window (5 frames @ 10fps, 12 @ 30fps)

# Stage 3 (YOLO11-seg + EasyOCR) & Picture-in-Picture Hyperparameters
STAGE3_CONF = 0.30         # License plate segmentation confidence
OCR_READS_PER_SEC = 5.0    # Run EasyOCR ~5 times per second per bike
ONLY_OCR_VIOLATIONS = True  # Set True to only run Stage 3 on Red/Orange violation bikes
MIN_PLATE_WIDTH = 22       # Skip OCR if plate crop is narrower than 22px
OCR_LANGUAGES = ["en"]     # EasyOCR language list
PIP_HOLD_SEC = 2.5         # Keep the PiP zoomed plate overlay visible for 2.5 seconds
MAX_PIP_CARDS = 2          # Maximum stacked PiP cards shown at once in the corner


# =====================================================================
# 2. HELPER FUNCTIONS (WEIGHTED VOTING, RECTIFICATION, OCR & PiP HUD)
# =====================================================================
def get_compute_device():
    """Selects NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU automatically."""
    if torch.cuda.is_available():
        return 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def natural_sort_key(path_obj):
    """Ensures frame '2.jpg' comes before '10.jpg'."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path_obj.name)
    ]


def convert_images_to_mp4(frame_paths, output_mp4_path: Path, fps: float = 10.0):
    """Stitches a sorted list of OSF image frames into a clean raw MP4 video."""
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first image: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4_path), fourcc, float(fps), (width, height))

    for img_path in tqdm(frame_paths, desc=f"Building Raw MP4 ({output_mp4_path.name})", leave=False):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)

    writer.release()
    return output_mp4_path


def select_weighted_label(label_history, conf_history):
    """
    5-Frame Confidence + Recency weighted voting with violation priority.
    Blocks 1-frame noise while snapping to clear close-up violations in 1-2 frames.
    """
    if not label_history:
        return "Safe", 0.0

    latest_label = label_history[-1]
    latest_conf = conf_history[-1]
    if latest_label != "Safe" and latest_conf >= 0.88 and len(label_history) <= 3:
        return latest_label, latest_conf

    class_scores = defaultdict(float)
    class_confs = defaultdict(list)

    n = len(label_history)
    for i, (lbl, conf) in enumerate(zip(label_history, conf_history)):
        recency_weight = 0.6 + 0.4 * (i / max(1, n - 1))
        violation_boost = 1.25 if lbl != "Safe" else 1.0
        class_scores[lbl] += conf * recency_weight * violation_boost
        class_confs[lbl].append(conf)

    best_label = max(class_scores.items(), key=lambda x: x[1])[0]
    avg_conf = float(np.mean(class_confs[best_label]))
    return best_label, avg_conf


def extract_and_rectify_plate(bike_crop: np.ndarray, polygon_pts: np.ndarray, box_xyxy: np.ndarray):
    """Deskews and crops the license plate using YOLO11-seg polygon points."""
    h, w = bike_crop.shape[:2]

    if polygon_pts is not None and len(polygon_pts) >= 4:
        pts = polygon_pts.astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        s = box.sum(axis=1)
        diff = np.diff(box, axis=1)
        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = box[np.argmin(s)]
        ordered[2] = box[np.argmax(s)]
        ordered[1] = box[np.argmin(diff)]
        ordered[3] = box[np.argmax(diff)]

        width_a = np.linalg.norm(ordered[2] - ordered[3])
        width_b = np.linalg.norm(ordered[1] - ordered[0])
        max_w = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(ordered[1] - ordered[2])
        height_b = np.linalg.norm(ordered[0] - ordered[3])
        max_h = max(int(height_a), int(height_b))

        if max_w >= MIN_PLATE_WIDTH and max_h >= 10:
            dst = np.array(
                [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(bike_crop, M, (max_w, max_h))
            return cv2.resize(warped, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    px1, py1, px2, py2 = box_xyxy.astype(int)
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w, px2), min(h, py2)
    plate_crop = bike_crop[py1:py2, px1:px2]
    if plate_crop.shape[1] < MIN_PLATE_WIDTH or plate_crop.shape[0] < 8:
        return None

    return cv2.resize(plate_crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)


def clean_plate_text(raw_text: str) -> str:
    """Retains only uppercase alphanumeric characters and hyphens."""
    cleaned = re.sub(r"[^A-Z0-9\-]", "", raw_text.upper())
    return cleaned if len(cleaned) >= 3 else ""


def select_best_plate(readings_list):
    """Locks in the highest-scoring license plate string for a tracked motorcycle."""
    if not readings_list:
        return None, 0.0
    score_by_text = defaultdict(float)
    best_conf_by_text = defaultdict(float)
    for text, conf in readings_list:
        score_by_text[text] += conf
        best_conf_by_text[text] = max(best_conf_by_text[text], conf)

    best_text = max(score_by_text.items(), key=lambda x: x[1])[0]
    return best_text, best_conf_by_text[best_text]


def draw_pip_violation_cards(frame: np.ndarray, active_cards: list, class_colors: dict):
    """
    Renders a Picture-in-Picture (PiP) overlay in the top-right corner of the frame
    showing the zoomed-in segmented license plate and EasyOCR text for active violations.
    """
    if not active_cards:
        return

    frame_h, frame_w = frame.shape[:2]
    scale = max(0.65, min(1.5, frame_w / 1920.0))

    card_w = int(340 * scale)
    header_h = int(34 * scale)
    plate_slot_h = int(115 * scale)
    footer_h = int(42 * scale)
    card_h = header_h + plate_slot_h + footer_h
    margin = int(20 * scale)
    pad = int(10 * scale)

    font = cv2.FONT_HERSHEY_SIMPLEX
    header_fs = 0.55 * scale
    footer_fs = 0.65 * scale
    thickness = max(1, int(round(2 * scale)))

    for idx, card in enumerate(active_cards[:MAX_PIP_CARDS]):
        x1 = frame_w - card_w - margin
        y1 = margin + idx * (card_h + margin)
        x2 = x1 + card_w
        y2 = y1 + card_h

        # Safety bounds check for narrow or portrait videos
        if x1 < 0 or y2 > frame_h:
            continue

        viol_label = card["violation"]
        accent_color = class_colors.get(viol_label, (0, 0, 220))

        # 1. Semi-transparent dark background panel
        roi = frame[y1:y2, x1:x2]
        dark_panel = np.full_like(roi, (24, 24, 28))
        cv2.addWeighted(dark_panel, 0.85, roi, 0.15, 0, roi)
        frame[y1:y2, x1:x2] = roi

        # 2. Colored top banner with Track ID & Violation Type
        cv2.rectangle(frame, (x1, y1), (x2, y1 + header_h), accent_color, -1)
        header_text = f"ID:{card['track_id']} | {viol_label}"
        cv2.putText(
            frame,
            header_text,
            (x1 + pad, y1 + int(header_h * 0.72)),
            font,
            header_fs,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # 3. Zoomed-in Segmented License Plate Inset (Aspect-Ratio Preserved)
        plate_img = card["plate_img"]
        if plate_img is not None and plate_img.size > 0:
            avail_w = card_w - 2 * pad
            avail_h = plate_slot_h - 2 * pad
            ph, pw = plate_img.shape[:2]

            resize_ratio = min(avail_w / max(1, pw), avail_h / max(1, ph))
            new_w = max(1, int(pw * resize_ratio))
            new_h = max(1, int(ph * resize_ratio))

            zoomed_plate = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            px1 = x1 + (card_w - new_w) // 2
            py1 = y1 + header_h + (plate_slot_h - new_h) // 2
            px2 = px1 + new_w
            py2 = py1 + new_h

            frame[py1:py2, px1:px2] = zoomed_plate
            # Cyan border around the zoomed license plate crop
            cv2.rectangle(frame, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), (255, 255, 0), 2)

        # 4. Footer Bar with EasyOCR Text & Confidence
        fy1 = y1 + header_h + plate_slot_h
        cv2.rectangle(frame, (x1, fy1), (x2, y2), (15, 15, 18), -1)

        plate_str = card.get("plate_text")
        ocr_conf = card.get("ocr_conf", 0.0)
        if plate_str:
            ocr_caption = f"PLATE: {plate_str} ({ocr_conf * 100:.0f}%)"
            text_color = (0, 255, 255)  # High-contrast Yellow/Cyan
        else:
            ocr_caption = "PLATE: SCANNING..."
            text_color = (190, 190, 190)

        cv2.putText(
            frame,
            ocr_caption,
            (x1 + pad, fy1 + int(footer_h * 0.68)),
            font,
            footer_fs,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        # Outer border around the entire PiP card
        cv2.rectangle(frame, (x1, y1), (x2, y2), accent_color, max(2, thickness))


# =====================================================================
# 3. THREE-STAGE INFERENCE PIPELINE
# =====================================================================
def run_three_stage_inference(
    clip_name: str,
    input_mp4_path: Path,
    output_mp4_path: Path,
    tickets_dir: Path,
    detector: YOLO,
    classifier: YOLO,
    plate_segmenter: YOLO,
    ocr_reader: easyocr.Reader,
    device,
    csv_writer,
):
    CLASS_COLORS = {
        "Safe": (34, 139, 34),             # Forest Green (BGR)
        "Helmet_Violation": (0, 0, 220),   # Crimson Red (BGR)
        "3Plus_Violation": (0, 140, 255),  # High-Vis Orange (BGR)
    }
    PLATE_COLOR = (255, 255, 0)            # Cyan highlight for segmented plate

    cap = cv2.VideoCapture(str(input_mp4_path))
    if not cap.isOpened():
        print(f"❌ Could not open video file: {input_mp4_path}")
        return

    # Read actual decoded first frame to guarantee true width/height on rotated mobile videos
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        print(f"❌ Could not decode frames from: {input_mp4_path}")
        return

    height, width = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Auto-detect native FPS from video header
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    if detected_fps is None or detected_fps <= 1.0 or detected_fps > 240.0 or np.isnan(detected_fps):
        fps = DEFAULT_OSF_FPS
    else:
        fps = float(detected_fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dynamically scale voting window, OCR stride, and PiP hold duration to native FPS
    voting_window = max(5, int(round(fps * VOTING_DURATION_SEC)))
    ocr_stride = max(1, int(round(fps / OCR_READS_PER_SEC)))
    pip_hold_frames = max(10, int(round(fps * PIP_HOLD_SEC)))

    # Scale label text and bounding box thickness for 1080p vs 4K mobile footage
    scale_factor = max(width, height) / 1920.0
    font_scale = max(0.55, 0.65 * scale_factor)
    box_thickness = max(2, int(round(3 * scale_factor)))
    font_thickness = max(1, int(round(2 * scale_factor)))

    print(
        f"   📊 Video Specs: {width}x{height} @ {fps:.2f} FPS | "
        f"Weighted Window: {voting_window}f | OCR Stride: {ocr_stride}f | PiP Hold: {pip_hold_frames}f"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (width, height))

    # Reset ByteTrack internal state so IDs start fresh at 1 for each new video
    if hasattr(detector, "predictor") and detector.predictor is not None:
        detector.predictor.trackers = []

    track_history = defaultdict(lambda: deque(maxlen=voting_window))
    track_conf_history = defaultdict(lambda: deque(maxlen=voting_window))
    track_plate_history = defaultdict(lambda: deque(maxlen=25))
    logged_tickets = {}

    # Stores the Picture-in-Picture state per offending track_id
    pip_tracker_state = {}

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"Inference ({output_mp4_path.name})")
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        clean_frame = frame.copy()

        # -----------------------------------------------------------------
        # STAGE 1: Detect & Track Motorcycles (YOLO26s Class 0 = motorcycle)
        # -----------------------------------------------------------------
        results = detector.track(
            frame,
            persist=True,
            classes=[0],
            conf=STAGE1_CONF,
            iou=STAGE1_IOU,
            tracker="bytetrack.yaml",
            device=device,
            verbose=False,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.astype(int)

                # +18% Dilation for Stage 2 Classification
                pad_w = int((x2 - x1) * BOX_DILATION)
                pad_h = int((y2 - y1) * BOX_DILATION)

                crop_x1 = max(0, x1 - pad_w)
                crop_y1 = max(0, y1 - pad_h)
                crop_x2 = min(width, x2 + pad_w)
                crop_y2 = min(height, y2 + pad_h)

                bike_crop = clean_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if bike_crop.size == 0:
                    continue

                # ---------------------------------------------------------
                # STAGE 2: Classify Violation (5-Frame Weighted Voting)
                # ---------------------------------------------------------
                cls_res = classifier.predict(
                    source=bike_crop,
                    imgsz=STAGE2_IMGSZ,
                    device=device,
                    verbose=False,
                )[0]
                pred_label = cls_res.names[cls_res.probs.top1]
                confidence = float(cls_res.probs.top1conf.cpu().item())

                track_history[track_id].append(pred_label)
                track_conf_history[track_id].append(confidence)

                voted_label, avg_conf = select_weighted_label(
                    track_history[track_id], track_conf_history[track_id]
                )

                # ---------------------------------------------------------
                # STAGE 3: License Plate Segmentation (YOLO11-seg) + EasyOCR
                # ---------------------------------------------------------
                should_run_stage3 = (not ONLY_OCR_VIOLATIONS) or (voted_label != "Safe")
                latest_plate_crop = None

                if should_run_stage3 and plate_segmenter is not None:
                    seg_res = plate_segmenter.predict(
                        source=bike_crop,
                        conf=STAGE3_CONF,
                        device=device,
                        verbose=False,
                    )[0]

                    if seg_res.boxes is not None and len(seg_res.boxes) > 0:
                        best_idx = int(seg_res.boxes.conf.argmax().item())
                        p_box = seg_res.boxes.xyxy[best_idx].cpu().numpy()

                        poly_pts = None
                        if seg_res.masks is not None and len(seg_res.masks.xy) > best_idx:
                            poly_pts = seg_res.masks.xy[best_idx]

                        # Draw cyan license plate segmentation contour on main frame
                        if poly_pts is not None and len(poly_pts) >= 3:
                            global_poly = (poly_pts + np.array([crop_x1, crop_y1])).astype(np.int32)
                            cv2.polylines(
                                frame, [global_poly], isClosed=True, color=PLATE_COLOR, thickness=box_thickness
                            )
                        else:
                            gx1, gy1, gx2, gy2 = (
                                p_box + np.array([crop_x1, crop_y1, crop_x1, crop_y1])
                            ).astype(int)
                            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), PLATE_COLOR, box_thickness)

                        # Rectify/crop the segmented plate for PiP display & EasyOCR
                        latest_plate_crop = extract_and_rectify_plate(bike_crop, poly_pts, p_box)

                        # Run EasyOCR on the FPS-adapted stride (or immediately if no plate read yet)
                        needs_initial_read = len(track_plate_history[track_id]) == 0
                        if latest_plate_crop is not None and ((frame_idx % ocr_stride == 0) or needs_initial_read):
                            ocr_outputs = ocr_reader.readtext(latest_plate_crop, detail=1)
                            if ocr_outputs:
                                combined_raw = "".join([item[1] for item in ocr_outputs])
                                mean_ocr_conf = float(np.mean([item[2] for item in ocr_outputs]))
                                cleaned_text = clean_plate_text(combined_raw)

                                if cleaned_text and mean_ocr_conf >= 0.25:
                                    track_plate_history[track_id].append((cleaned_text, mean_ocr_conf))

                                    if voted_label != "Safe":
                                        prev_best = logged_tickets.get(track_id, (None, 0.0))[1]
                                        if mean_ocr_conf > prev_best:
                                            ticket_img_name = (
                                                f"{clip_name}_ID{track_id}_{voted_label}_{cleaned_text}.jpg"
                                            )
                                            cv2.imwrite(str(tickets_dir / ticket_img_name), bike_crop)
                                            logged_tickets[track_id] = (
                                                cleaned_text,
                                                mean_ocr_conf,
                                                voted_label,
                                                frame_idx,
                                                ticket_img_name,
                                            )

                best_plate, best_plate_conf = select_best_plate(track_plate_history[track_id])

                # ---------------------------------------------------------
                # UPDATE PICTURE-IN-PICTURE STATE WHEN VIOLATION IS DETECTED
                # ---------------------------------------------------------
                if voted_label != "Safe" and (latest_plate_crop is not None or track_id in pip_tracker_state):
                    existing = pip_tracker_state.get(track_id)

                    # Keep the sharpest/highest-confidence plate crop for the PiP window
                    chosen_crop = existing["plate_img"] if existing is not None else latest_plate_crop
                    if latest_plate_crop is not None:
                        if existing is None or chosen_crop is None:
                            chosen_crop = latest_plate_crop
                        elif best_plate_conf >= existing.get("ocr_conf", 0.0):
                            chosen_crop = latest_plate_crop

                    if chosen_crop is not None:
                        pip_tracker_state[track_id] = {
                            "track_id": int(track_id),
                            "violation": voted_label,
                            "plate_img": chosen_crop,
                            "plate_text": best_plate,
                            "ocr_conf": best_plate_conf,
                            "last_seen_frame": frame_idx,
                        }

                # ---------------------------------------------------------
                # DRAW BOUNDING BOX & LABEL ON MAIN FRAME
                # ---------------------------------------------------------
                color = CLASS_COLORS.get(voted_label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

                if best_plate:
                    caption = f"ID:{track_id} | {voted_label} ({avg_conf * 100:.0f}%) | Plate: {best_plate}"
                else:
                    caption = f"ID:{track_id} | {voted_label} ({avg_conf * 100:.0f}%)"

                (txt_w, txt_h), _ = cv2.getTextSize(
                    caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                label_y1 = max(0, y1 - txt_h - 12)
                label_y2 = max(txt_h + 12, y1)

                cv2.rectangle(frame, (x1, label_y1), (x1 + txt_w + 10, label_y2), color, -1)
                cv2.putText(
                    frame,
                    caption,
                    (x1 + 5, label_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )

        # -----------------------------------------------------------------
        # RENDER PICTURE-IN-PICTURE ZOOMED PLATE OVERLAY IN TOP-RIGHT CORNER
        # -----------------------------------------------------------------
        active_pip_cards = [
            state
            for state in pip_tracker_state.values()
            if (frame_idx - state["last_seen_frame"]) <= pip_hold_frames
        ]
        # Sort so the most recently seen violation appears at the top
        active_pip_cards.sort(key=lambda s: (s["last_seen_frame"], s["ocr_conf"]), reverse=True)
        draw_pip_violation_cards(frame, active_pip_cards, CLASS_COLORS)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    for track_id, (plate_str, ocr_conf, viol_type, f_idx, img_name) in logged_tickets.items():
        csv_writer.writerow([clip_name, track_id, viol_type, plate_str, f"{ocr_conf:.2f}", f_idx, img_name])


# =====================================================================
# 4. MAIN EXECUTION (DISCOVERS BOTH MOBILE VIDEOS & OSF IMAGE FOLDERS)
# =====================================================================
def main():
    input_dir = Path(INPUT_MEDIA_DIR)
    output_dir = Path(OUTPUT_VIDEOS_DIR)
    raw_dir = output_dir / "raw_mp4"
    annotated_dir = output_dir / "annotated_mp4"
    tickets_dir = output_dir / "violation_tickets"

    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    tickets_dir.mkdir(parents=True, exist_ok=True)

    if not Path(STAGE1_WEIGHTS).exists():
        raise FileNotFoundError(f"Stage 1 weights not found at: {STAGE1_WEIGHTS}")
    if not Path(STAGE2_WEIGHTS).exists():
        raise FileNotFoundError(f"Stage 2 weights not found at: {STAGE2_WEIGHTS}")

    device = get_compute_device()
    use_gpu_ocr = device == 0 or device == "mps"
    print(f"🖥️  Using Compute Device: {device}")

    print(f"🎯 Loading Stage 1 Detector:   {STAGE1_WEIGHTS}")
    detector = YOLO(STAGE1_WEIGHTS)

    print(f"🧠 Loading Stage 2 Classifier: {STAGE2_WEIGHTS}")
    classifier = YOLO(STAGE2_WEIGHTS)

    plate_segmenter = None
    ocr_reader = None
    if Path(STAGE3_PLATE_WEIGHTS).exists():
        print(f"🔍 Loading Stage 3 YOLO11-seg: {STAGE3_PLATE_WEIGHTS}")
        plate_segmenter = YOLO(STAGE3_PLATE_WEIGHTS)
        print(f"📖 Initializing EasyOCR Reader ({OCR_LANGUAGES}, gpu={use_gpu_ocr})...")
        ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu_ocr)
    else:
        print(
            f"⚠️  Stage 3 weight not found at {STAGE3_PLATE_WEIGHTS}.\n"
            "   Running Stage 1 + Stage 2 only until you place the Stage 3 weight in weights/."
        )

    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png"}

    media_queue = []
    for item in sorted(input_dir.iterdir()):
        if TARGET_CLIPS and item.stem not in TARGET_CLIPS and item.name not in TARGET_CLIPS:
            continue
        if item.is_file() and item.suffix.lower() in video_exts:
            media_queue.append((item.stem, "video", item))
        elif item.is_dir():
            frames = sorted(
                [p for p in item.iterdir() if p.suffix.lower() in image_exts],
                key=natural_sort_key,
            )
            if frames:
                media_queue.append((item.name, "image_folder", frames))

    if not media_queue:
        print(f"\n⚠️  No video files or image folders found inside {input_dir}.")
        return

    print(f"\n🎬 Found {len(media_queue)} item(s) to process:\n" + "=" * 65)

    csv_log_path = tickets_dir / "violation_tickets_log.csv"
    with open(csv_log_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["Video_Clip", "Track_ID", "Violation_Class", "License_Plate", "OCR_Confidence", "Best_Frame", "Evidence_Image"]
        )

        for clip_name, media_type, source_data in media_queue:
            if media_type == "image_folder":
                print(f"\n▶️  Processing OSF Image Folder '{clip_name}' ({len(source_data)} frames @ {DEFAULT_OSF_FPS} FPS)")
                source_mp4_path = raw_dir / f"{clip_name}_raw.mp4"
                convert_images_to_mp4(source_data, source_mp4_path, fps=DEFAULT_OSF_FPS)
            else:
                print(f"\n▶️  Processing Mobile/Direct Video '{source_data.name}'")
                source_mp4_path = source_data

            annotated_mp4_path = annotated_dir / f"{clip_name}_annotated.mp4"
            run_three_stage_inference(
                clip_name=clip_name,
                input_mp4_path=source_mp4_path,
                output_mp4_path=annotated_mp4_path,
                tickets_dir=tickets_dir,
                detector=detector,
                classifier=classifier,
                plate_segmenter=plate_segmenter,
                ocr_reader=ocr_reader,
                device=device,
                csv_writer=csv_writer,
            )
            print(f"   ✅ Annotated video saved: {annotated_mp4_path}")

    print(f"\n🎉 All videos completed! Violation ticket log saved to:\n   {csv_log_path}")


if __name__ == "__main__":
    main()