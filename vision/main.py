
import os
import time
import cv2
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# 1. LOAD MODELS
# --------------------------------------------------

# Standard COCO model for normal vehicle counting
COCO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# Custom-trained ambulance model
AMBULANCE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs",
    "ambulance_experiments",
    "clean_labels_100e",
    "weights",
    "best.pt",
)

if not os.path.isfile(COCO_MODEL_PATH):
    raise FileNotFoundError(
        f"COCO model not found: {COCO_MODEL_PATH}"
    )

if not os.path.isfile(AMBULANCE_MODEL_PATH):
    raise FileNotFoundError(
        f"Ambulance model not found: {AMBULANCE_MODEL_PATH}"
    )

coco_model = YOLO(COCO_MODEL_PATH)
ambulance_model = YOLO(AMBULANCE_MODEL_PATH)

print("COCO model loaded:", COCO_MODEL_PATH)
print("Ambulance model loaded:", AMBULANCE_MODEL_PATH)
print("Ambulance model classes:", ambulance_model.names)

# The custom model uses:
# Class 0 = Ambulance
# Class 1 = not Ambulance
AMBULANCE_CLASS_ID = 0

# Detection confidence thresholds
EMERGENCY_CONF_THRESH = 0.50
COCO_CONF_THRESH = 0.25

# COCO class IDs: car, motorcycle, bus, truck
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]

# --------------------------------------------------
# 2. VIDEO SOURCES AND TRAFFIC SETTINGS
# --------------------------------------------------

LANE_VIDEOS = {
    "Lane 1": os.path.join(BASE_DIR, "lane1.mp4"),
    "Lane 2": os.path.join(BASE_DIR, "lane2.mp4"),
    "Lane 3": os.path.join(BASE_DIR, "lane3.mp4"),
    "Lane 4": os.path.join(BASE_DIR, "lane4.mp4"),
}

LANES_ORDER = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]

EMERGENCY_CONFIRM_FRAMES = 3

# Shared traffic state
system_state = {
    "active_green_lane": "Lane 1",
    "emergency_detected": False,
    "emergency_lane": None,
    "current_timer": 15,
    "is_manual": False,
    "lane_data": {
        lane: {
            "emergency_count": 0,
            "normal_count": 0,
        }
        for lane in LANES_ORDER
    },
}

# Consecutive frames with ambulance detection
emergency_streak = {
    lane: 0 for lane in LANES_ORDER
}

# Lanes waiting for emergency priority
emergency_queue = []

manual_override_lane = None

# Protect shared state between threads
state_lock = threading.RLock()


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def calculate_iou(box1, box2):
    """Calculate overlap between two bounding boxes."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) * max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_green_time(vehicle_count):
    """Allocate green time according to normal vehicle count."""

    if vehicle_count < 10:
        return 15
    elif vehicle_count <= 20:
        return 30
    else:
        return 45


def get_next_sequential_lane(current_lane):
    """Return the next lane in the configured sequence."""

    idx = LANES_ORDER.index(current_lane)
    return LANES_ORDER[(idx + 1) % len(LANES_ORDER)]


def add_emergency_to_queue(lane):
    """Add a lane to the queue once, without duplicates."""

    if lane not in emergency_queue:
        emergency_queue.append(lane)
        print(f"[EMERGENCY] {lane} added to priority queue.")


# --------------------------------------------------
# 4. TRAFFIC SIGNAL CONTROLLER
# --------------------------------------------------

def traffic_controller():
    """Manage automatic switching and emergency priority."""

    global manual_override_lane

    while True:
        # Handle manual override
        with state_lock:
            is_manual = system_state["is_manual"]
            override_lane = manual_override_lane

        if is_manual and override_lane:
            with state_lock:
                system_state["active_green_lane"] = override_lane
                system_state["current_timer"] = 30

            time.sleep(1)
            continue

        # Get current lane and its vehicle count
        with state_lock:
            active_lane = system_state["active_green_lane"]
            vehicle_count = system_state["lane_data"][
                active_lane
            ]["normal_count"]

        allocated_time = calculate_green_time(vehicle_count)
        interrupted = False

        # Countdown for the active green lane
        for remaining in range(allocated_time, 0, -1):
            with state_lock:
                if system_state["is_manual"]:
                    interrupted = True
                    break

                system_state["current_timer"] = remaining

            time.sleep(1)

        if interrupted:
            continue

        # Choose the next lane
        with state_lock:
            if system_state["is_manual"]:
                continue

            # Ignore stale queue entries if their emergency cleared
            emergency_queue[:] = [
                lane
                for lane in emergency_queue
                if system_state["lane_data"][lane][
                    "emergency_count"
                ] > 0
            ]

            if emergency_queue:
                next_lane = emergency_queue.pop(0)

                system_state["emergency_detected"] = True
                system_state["emergency_lane"] = next_lane
                system_state["active_green_lane"] = next_lane

                print(
                    f"[PRIORITY] Green signal assigned to "
                    f"{next_lane}."
                )

            else:
                system_state["emergency_detected"] = False
                system_state["emergency_lane"] = None

                next_lane = get_next_sequential_lane(active_lane)
                system_state["active_green_lane"] = next_lane

            system_state["current_timer"] = 15


# --------------------------------------------------
# 5. VIDEO PROCESSING AND DETECTION
# --------------------------------------------------

def process_video_feeds():
    """Detect ambulances and vehicles in all available lane videos."""

    caps = {}

    # Open available videos
    for lane, path in LANE_VIDEOS.items():
        if not os.path.isfile(path):
            print(f"[WARNING] Video missing for {lane}: {path}")
            continue

        cap = cv2.VideoCapture(path)

        if cap.isOpened():
            caps[lane] = cap
            print(f"[VIDEO] Opened {lane}: {path}")
        else:
            print(f"[ERROR] Could not open video for {lane}: {path}")
            cap.release()

    if not caps:
        print(
            "[ERROR] No usable lane videos found. "
            "Check the video filenames and folder."
        )
        return

    frame_number = 0

    try:
        while True:
            for lane, cap in caps.items():
                ret, frame = cap.read()

                # Restart video when it reaches the end
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                    if not ret:
                        print(
                            f"[WARNING] Could not read a frame "
                            f"from {lane}."
                        )
                        continue

                frame_number += 1

                # ------------------------------------------
                # A. CUSTOM AMBULANCE DETECTION
                # ------------------------------------------

                amb_results = ambulance_model.predict(
                    source=frame,
                    conf=EMERGENCY_CONF_THRESH,
                    verbose=False,
                )[0]

                emergency_boxes = []

                if amb_results.boxes is not None:
                    for box in amb_results.boxes:
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        # Only class 0 is an ambulance.
                        if (
                            cls_id == AMBULANCE_CLASS_ID
                            and confidence >= EMERGENCY_CONF_THRESH
                        ):
                            coords = [
                                int(value)
                                for value in box.xyxy[0].tolist()
                            ]

                            emergency_boxes.append(
                                (coords, confidence)
                            )

                emergency_count = len(emergency_boxes)

                # ------------------------------------------
                # B. NORMAL VEHICLE DETECTION
                # ------------------------------------------

                coco_results = coco_model.predict(
                    source=frame,
                    conf=COCO_CONF_THRESH,
                    classes=COCO_VEHICLE_CLASSES,
                    verbose=False,
                )[0]

                normal_count = 0
                coco_boxes = []

                if coco_results.boxes is not None:
                    for box in coco_results.boxes:
                        coords = [
                            int(value)
                            for value in box.xyxy[0].tolist()
                        ]

                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        # Avoid counting a vehicle detection again
                        # if it overlaps a custom ambulance box.
                        overlaps_ambulance = any(
                            calculate_iou(
                                coords,
                                ambulance_box,
                            ) > 0.20
                            for ambulance_box, _ in emergency_boxes
                        )

                        if not overlaps_ambulance:
                            coco_boxes.append(
                                (coords, cls_id, confidence)
                            )
                            normal_count += 1

                # ------------------------------------------
                # C. UPDATE SHARED LANE DATA
                # ------------------------------------------

                with state_lock:
                    system_state["lane_data"][lane] = {
                        "emergency_count": (
                            1 if emergency_count > 0 else 0
                        ),
                        "normal_count": normal_count,
                    }

                    if emergency_count > 0:
                        emergency_streak[lane] += 1

                        if (
                            emergency_streak[lane]
                            >= EMERGENCY_CONFIRM_FRAMES
                            and lane
                            != system_state["active_green_lane"]
                        ):
                            add_emergency_to_queue(lane)

                    else:
                        emergency_streak[lane] = 0

                        # Remove the lane if its emergency has cleared.
                        if lane in emergency_queue:
                            emergency_queue.remove(lane)
                            print(
                                f"[EMERGENCY] Cleared from {lane}; "
                                f"removed from queue."
                            )

                # ------------------------------------------
                # D. DRAW NORMAL VEHICLE BOXES
                # ------------------------------------------

                display_frame = frame.copy()

                for coords, cls_id, confidence in coco_boxes:
                    x1, y1, x2, y2 = coords

                    vehicle_name = coco_model.names.get(
                        cls_id, "vehicle"
                    )

                    label = f"{vehicle_name} {confidence:.2f}"

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (220, 220, 220),
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        label,
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (220, 220, 220),
                        2,
                    )

                # ------------------------------------------
                # E. DRAW RED AMBULANCE BOXES
                # ------------------------------------------

                for coords, confidence in emergency_boxes:
                    x1, y1, x2, y2 = coords

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3,
                    )

                    cv2.putText(
                        display_frame,
                        f"AMBULANCE {confidence:.2f}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # ------------------------------------------
                # F. SHOW LANE STATUS ON VIDEO
                # ------------------------------------------

                with state_lock:
                    active_lane = system_state["active_green_lane"]
                    is_manual = system_state["is_manual"]

                status = (
                    "GREEN SIGNAL"
                    if lane == active_lane
                    else "WAITING"
                )

                cv2.putText(
                    display_frame,
                    f"{lane}: {status}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if lane == active_lane
                    else (255, 255, 255),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"Vehicles: {normal_count} | "
                    f"Ambulances: {emergency_count}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

                if is_manual:
                    cv2.putText(
                        display_frame,
                        "MANUAL OVERRIDE",
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )

                resized_frame = cv2.resize(
                    display_frame,
                    (480, 270),
                )

                cv2.imshow(
                    f"Live Traffic Feed - {lane}",
                    resized_frame,
                )

            # Press Q in an OpenCV window to close the feeds.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as exc:
        print(f"[ERROR] Video processing stopped: {exc}")
        raise

    finally:
        for cap in caps.values():
            cap.release()

        cv2.destroyAllWindows()
        print("[VIDEO] All video feeds released.")


# --------------------------------------------------
# 6. API ENDPOINTS
# --------------------------------------------------

@app.route("/api/traffic", methods=["GET"])
def get_traffic_data():
    """Return current traffic and signal information."""

    with state_lock:
        return jsonify(system_state)


@app.route("/api/manual", methods=["POST", "OPTIONS"])
def manual_control():
    """Enable a manual green signal or resume automatic mode."""

    global manual_override_lane

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or {}

    raw_lane = str(
        data.get("lane")
        or data.get("lane_id")
        or data.get("lane_number")
        or ""
    ).strip()

    action = str(data.get("action", "enable")).lower().strip()

    # Resume automatic traffic control
    if action == "disable":
        with state_lock:
            system_state["is_manual"] = False
            manual_override_lane = None

        print("[MANUAL] Manual override disabled.")

        return jsonify({
            "status": "success",
            "message": "Manual override disabled.",
        }), 200

    # Normalize lane inputs such as "2", "Lane 2", or "lane 2"
    if raw_lane.isdigit():
        lane = f"Lane {raw_lane}"
    elif raw_lane.lower().startswith("lane"):
        lane = f"Lane {raw_lane.split()[-1]}"
    else:
        lane = raw_lane

    if lane not in LANES_ORDER:
        return jsonify({
            "status": "error",
            "message": f"Invalid lane: {lane}",
        }), 400

    with state_lock:
        system_state["is_manual"] = True
        manual_override_lane = lane
        system_state["active_green_lane"] = lane
        system_state["current_timer"] = 30

    print(f"[MANUAL] Green signal forced to {lane}.")

    return jsonify({
        "status": "success",
        "message": f"Manual override set to {lane}",
    }), 200


# --------------------------------------------------
# 7. START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    video_thread = threading.Thread(
        target=process_video_feeds,
        daemon=True,
    )
    video_thread.start()

    controller_thread = threading.Thread(
        target=traffic_controller,
        daemon=True,
    )
    controller_thread.start()

    print("Smart Traffic System backend starting on port 5000.")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
