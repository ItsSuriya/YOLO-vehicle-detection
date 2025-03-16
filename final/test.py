import cv2
import torch
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pymongo import MongoClient
from datetime import datetime

# Set device with CUDA priority
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# MongoDB Atlas connection
MONGO_URI = "mongodb+srv://AI_agent:z8W1L0n41kZvseDw@unisys.t75li.mongodb.net/?retryWrites=true&w=majority&appName=Unisys"
client = MongoClient(MONGO_URI)
db = client["vehicle_detection"]  # Database name
collection = db["detected_vehicles"]  # Collection name

# Class definitions
Modal_classes = ['MPV', 'Minivan']
Vehicle_classes = ['Bus', 'Car', 'Two wheeler']
Number_plate_classes = ['Number Plate']

# Color detection parameters
COLOR_RANGES = {
    'red':    ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
    'blue':   ([100, 150, 50], [140, 255, 255]),
    'green':  ([40, 50, 50], [80, 255, 255]),
    'white':  ([0, 0, 200], [180, 30, 255]),
    'black':  ([0, 0, 0], [180, 255, 30]),
    'yellow': ([20, 100, 100], [40, 255, 255]),
    'silver': ([0, 0, 150], [180, 30, 210]),
    'gray':   ([0, 0, 50], [180, 30, 150]),
    'orange': ([10, 100, 100], [20, 255, 255]),
}

def detect_vehicle_color(roi):
    """Detect dominant vehicle color using HSV histogram analysis"""
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
        
        hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        dominant_hue = np.argmax(hist)
        
        color_scores = {}
        for color, ranges in COLOR_RANGES.items():
            if len(ranges) == 4:
                lower1, upper1, lower2, upper2 = ranges
                mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                score = (cv2.countNonZero(mask1) + cv2.countNonZero(mask2)) / (roi.size / 3)
            else:
                lower, upper = ranges
                score = cv2.countNonZero(cv2.inRange(hsv, np.array(lower), np.array(upper))) / (roi.size / 3)
            
            color_scores[color] = score
        
        best_color = max(color_scores, key=color_scores.get)
        confidence = color_scores[best_color]
        return best_color, confidence
    except:
        return 'unknown', 0.0

# Initialize OCR reader with GPU support
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())

# Load models with CUDA optimization
def load_model(path):
    model = YOLO(path).to(device)
    model.fuse()
    if device == 'cuda':
        model.half()
    return model

vehicle_model = load_model(r"E:\Unisys project\Dataset\runs\detect\Vehicle-info2\weights\best.pt")
modal_model = load_model(r"E:\Unisys project\Dataset\NEW\runs\detect\last-info\weights\best.pt")
number_plate_model = load_model(r"E:\Unisys project\Dataset\runs\detect\Vehicle-Number plate\weights\best.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Dictionary to store vehicle information
vehicle_info = {}

def validate_roi(roi):
    """Check if ROI is valid and non-empty"""
    return roi is not None and roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10

def save_to_mongodb(track_id, data):
    """Save vehicle data to MongoDB with upsert operation"""
    try:
        collection.update_one(
            {"track_id": track_id},
            {"$set": {
                **data,
                "last_updated": datetime.now()
            }},
            upsert=True
        )
    except Exception as e:
        print(f"Database error: {e}")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = video_path.replace(".mp4", "_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    cv2.namedWindow('Vehicle Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Vehicle Tracking', 960, 540)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Vehicle detection with CUDA
        vehicle_results = vehicle_model.predict(frame, conf=0.6, verbose=False, device=device)
        
        # Prepare detections for DeepSORT
        detections = []
        for result in vehicle_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

        # Update tracker
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        # Process tracked vehicles
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Initialize vehicle info if new track
            if track_id not in vehicle_info:
                class_id = int(track.det_class) if track.det_class is not None else 0
                conf = float(track.det_conf) if track.det_conf is not None else 0.0
                
                vehicle_info[track_id] = {
                    'vehicle_class': Vehicle_classes[class_id],
                    'vehicle_confidence': conf,
                    'modal_class': "Unknown",
                    'modal_confidence': 0.0,
                    'color': "Unknown",
                    'color_confidence': 0.0,
                    'plate_confidence': 0.0,
                    'plate_text': "Not detected",
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
            else:
                vehicle_info[track_id]['last_seen'] = datetime.now()

            info = vehicle_info[track_id]
            
            # Draw vehicle tracking info
            label = f"ID:{track_id} {info['vehicle_class']} {info['vehicle_confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Process vehicle ROI
            vehicle_roi = frame[y1:y2, x1:x2]
            if validate_roi(vehicle_roi):
                # Color detection
                color, color_conf = detect_vehicle_color(vehicle_roi)
                info['color'] = color
                info['color_confidence'] = color_conf
                cv2.putText(frame, f"Color: {color} ({color_conf:.2f})", 
                           (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Modal detection
                try:
                    modal_results = modal_model.predict(vehicle_roi, conf=0.5, verbose=False, device=device)
                    for modal in modal_results:
                        for mbox, mcls, mconf in zip(modal.boxes.xyxy.cpu().numpy(),
                                                   modal.boxes.cls.cpu().numpy().astype(int),
                                                   modal.boxes.conf.cpu().numpy()):
                            mx1, my1, mx2, my2 = map(int, mbox)
                            info['modal_class'] = Modal_classes[int(mcls)]
                            info['modal_confidence'] = float(mconf)
                            cv2.rectangle(frame, (x1+mx1, y1+my1),
                                        (x1+mx2, y1+my2), (0, 255, 0), 2)
                            modal_label = f"{info['modal_class']} {info['modal_confidence']:.2f}"
                            cv2.putText(frame, modal_label, 
                                      (x1+mx1, y1+my1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Modal detection error: {e}")

                # Number plate detection
                try:
                    plate_results = number_plate_model.predict(vehicle_roi, conf=0.4, verbose=False, device=device)
                    for plate in plate_results:
                        for pbox, pcls, pconf in zip(plate.boxes.xyxy.cpu().numpy(),
                                                   plate.boxes.cls.cpu().numpy().astype(int),
                                                   plate.boxes.conf.cpu().numpy()):
                            px1, py1, px2, py2 = map(int, pbox)
                            info['plate_confidence'] = float(pconf)
                            
                            # OCR processing
                            plate_roi = vehicle_roi[py1:py2, px1:px2]
                            if validate_roi(plate_roi):
                                gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                                result = ocr_reader.ocr(gray_plate, cls=True)
                                if result and len(result[0]) > 0:
                                    info['plate_text'] = result[0][0][1][0]
                            
                            # Draw plate detection
                            cv2.rectangle(frame, (x1+px1, y1+py1),
                                        (x1+px2, y1+py2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Plate: {info['plate_text']}", 
                                      (x1+px1, y1+py2+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception as e:
                    print(f"Plate processing error: {e}")

                # Prepare data for MongoDB
                db_data = {
                    "track_id": track_id,
                    "vehicle_type": info['vehicle_class'],
                    "modal_type": info['modal_class'],
                    "color": info['color'],
                    "license_plate": info['plate_text'],
                    "confidence_scores": {
                        "vehicle": info['vehicle_confidence'],
                        "modal": info['modal_confidence'],
                        "color": info['color_confidence'],
                        "plate": info['plate_confidence']
                    },
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "timestamps": {
                        "first_seen": info['first_seen'],
                        "last_seen": info['last_seen']
                    }
                }
                
                # Save to MongoDB
                save_to_mongodb(track_id, db_data)

        # Write and display frame
        out.write(frame)
        cv2.imshow('Vehicle Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Final summary and bulk insert
    try:
        collection.insert_many([{
            **vehicle,
            "track_id": track_id,
            "final_update": datetime.now()
        } for track_id, vehicle in vehicle_info.items()])
    except Exception as e:
        print(f"Final database insert error: {e}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print final summary
    print("\nVehicle Tracking Summary:")
    for tid, data in vehicle_info.items():
        print(f"ID {tid}:")
        print(f"  Vehicle: {data['vehicle_class']} ({data['vehicle_confidence']:.2f})")
        print(f"  Type: {data['modal_class']} ({data['modal_confidence']:.2f})")
        print(f"  Color: {data['color']} ({data['color_confidence']:.2f})")
        print(f"  Plate: {data['plate_text']} ({data['plate_confidence']:.2f})")
        print("-" * 40)

# Process video
video_path = r"C:\Users\iamsu\Downloads\Untitled video - Made with Clipchamp (4).mp4"
process_video(video_path)
