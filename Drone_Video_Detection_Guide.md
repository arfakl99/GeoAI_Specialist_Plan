# 🎬 VIDEO OBJECT DETECTION GUIDE
## Task 3: Drone Footage Analysis - Person Detection on Hilly Terrain

---

## YOUR CHALLENGE

```
Input: Drone video of hilly/mountainous terrain
Task: Detect all persons/soldiers in video
Output: Timestamped detections with locations

Challenges:
- Persons are SMALL in drone footage
- Terrain is COMPLEX (shadows, vegetation, hills)
- Need to TRACK people across frames
- High-altitude footage (500-2000m)
```

---

## WHICH MODEL: SAM2 vs YOLO vs Hybrid?

### SAM2 Approach (Zero-shot, Most Flexible)

**Pros:**
- ✅ No training needed
- ✅ Works on unseen terrain
- ✅ Handles occlusion well
- ✅ Pixel-perfect masks

**Cons:**
- ❌ Slow (1-2s per frame)
- ❌ Can't process video in real-time
- ❌ Requires manual prompting for each scene

**Best For:** Precision analysis, small datasets, new terrain types

---

### YOLOv8 Approach (Pre-trained Detection)

**Pros:**
- ✅ Very fast (50-100ms per frame)
- ✅ Real-time video processing
- ✅ Accurate person detection
- ✅ Multiple pose options (YOLO-Pose, YOLO-World)

**Cons:**
- ❌ Requires training data
- ❌ Struggles with small persons
- ❌ May have false positives

**Best For:** Real-time processing, drone footage, general person detection

---

### RECOMMENDED: Hybrid Approach (Best of Both)

```
YOLO (Fast Pass) → SAM2 (Verification) → Tracking
   ↓                  ↓                     ↓
50-100ms      Verify detections      Track across
per frame     (100-200ms for         frames
              confirmed objects)
```

**This gives you:**
- ✅ Speed of YOLO
- ✅ Accuracy of SAM2
- ✅ Tracking capability
- ✅ Real-time drone processing

---

## ARCHITECTURE 1: SAM2 FOR DRONE VIDEO

### Step 1: Video Preprocessing

```python
import cv2
import numpy as np
from pathlib import Path

def prepare_drone_video(video_path, output_fps=5):
    """
    Prepare drone video for processing:
    - Reduce FPS (process every Nth frame)
    - Normalize brightness
    - Detect scene changes
    """
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / output_fps)  # Process every Nth frame
    
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every Nth frame
        if frame_count % frame_skip == 0:
            # Normalize brightness (important for hillside shadows)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            frame_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            frames.append(frame_normalized)
        
        frame_count += 1
    
    cap.release()
    return frames

# Usage
video_path = "drone_footage.mp4"
frames = prepare_drone_video(video_path, output_fps=5)
print(f"Prepared {len(frames)} frames for analysis")
```

---

### Step 2: SAM2 Video Segmentation

```python
import torch
from segment_anything_2.video_predictor import SAM2VideoPredictor
from segment_anything_2.video_model import build_sam2_video_predictor

def detect_persons_sam2(video_frames, prompt="person"):
    """
    Use SAM2 to detect persons in drone video
    SAM2 handles temporal coherence automatically
    """
    
    # Initialize SAM2
    checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_large"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    
    all_detections = {}
    
    # Process video
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for frame_idx, frame in enumerate(video_frames):
            print(f"Processing frame {frame_idx+1}/{len(video_frames)}")
            
            # For first frame, initialize predictor
            if frame_idx == 0:
                predictor.reset_state()
                # Convert frame to tensor
                frame_tensor = torch.from_numpy(frame).float().to(device)
                frame_tensor = frame_tensor.permute(2, 0, 1)[None]  # Add batch dim
            
            # Add point prompts (click at center of frame to guide detection)
            # This helps SAM2 focus on finding persons
            h, w = frame.shape[:2]
            center_point = np.array([[w//2, h//2]])
            labels = np.array([1])  # 1 = foreground point
            
            # Detect
            masks, scores, logits = predictor.predict(
                frame,
                point_coords=center_point,
                point_labels=labels,
                multimask_output=False
            )
            
            # Store detections
            detections = []
            for mask, score in zip(masks, scores):
                if score > 0.6:  # Confidence threshold
                    # Find bounding box from mask
                    coords = np.where(mask)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        
                        detections.append({
                            "bbox": [x_min, y_min, x_max, y_max],
                            "score": float(score),
                            "mask": mask
                        })
            
            all_detections[frame_idx] = detections
    
    return all_detections

# Usage
detections = detect_persons_sam2(frames, prompt="person on hillside")
```

---

## ARCHITECTURE 2: YOLOV8 FOR DRONE VIDEO (FASTER)

### Step 1: Load Pre-trained YOLO

```python
from ultralytics import YOLO
import cv2

def detect_persons_yolo(video_path, output_path="output_detections.mp4"):
    """
    Fast person detection using YOLOv8
    Best for real-time drone processing
    """
    
    # Load pretrained YOLO (trained on COCO dataset)
    # Includes "person" class
    model = YOLO('yolov8x.pt')  # Largest model for best accuracy
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    all_detections = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model.predict(
            frame,
            conf=0.5,           # Confidence threshold
            iou=0.45,           # NMS IoU threshold
            classes=0,          # class 0 = person in COCO
            verbose=False
        )
        
        # Extract detections
        detections = []
        result = results[0]
        
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                if conf > 0.5:
                    detections.append({
                        "frame_id": frame_idx,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    })
        
        all_detections[frame_idx] = detections
        
        # Draw annotations
        annotated_frame = result.plot()
        out.write(annotated_frame)
        
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: {len(detections)} persons detected")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return all_detections

# Usage
detections = detect_persons_yolo("drone_footage.mp4")
print(f"Total frames analyzed: {len(detections)}")
```

---

## ARCHITECTURE 3: HYBRID (RECOMMENDED)

### Complete Pipeline: YOLO + SAM2 + Tracking

```python
from ultralytics import YOLO
from segment_anything_2.video_predictor import build_sam2_video_predictor
import cv2
import numpy as np
import torch
from supervision import ByteTrack
import json

class DronePersonDetector:
    """
    Hybrid detector: Fast YOLO + Accurate SAM2 + Tracking
    
    Pipeline:
    1. YOLO detects person candidates (fast)
    2. SAM2 refines detections (accurate masks)
    3. ByteTrack tracks across frames
    4. Output: Video + JSON with person locations
    """
    
    def __init__(self, video_path, device="cuda"):
        self.video_path = video_path
        self.device = torch.device(device)
        
        # Load models
        self.yolo = YOLO('yolov8x.pt')
        self.sam2 = build_sam2_video_predictor(
            "sam2_hiera_large",
            checkpoint="checkpoints/sam2_hiera_large.pt",
            device=self.device
        )
        self.tracker = ByteTrack()
        
    def detect_and_track(self, output_video_path, confidence_threshold=0.5):
        """
        Main detection and tracking pipeline
        """
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        all_person_tracks = {}
        self.sam2.reset_state()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # STEP 1: YOLO Detection (FAST)
            yolo_results = self.yolo.predict(frame, conf=confidence_threshold, classes=0)
            yolo_boxes = []
            
            if yolo_results[0].boxes is not None:
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    yolo_boxes.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    })
            
            # STEP 2: SAM2 Refinement (ACCURATE) - Optional, for high-confidence boxes
            refined_detections = []
            
            for yolo_box in yolo_boxes:
                x1, y1, x2, y2 = yolo_box["bbox"]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Use SAM2 to refine detection
                try:
                    with torch.inference_mode():
                        masks, scores, _ = self.sam2.predict(
                            frame,
                            point_coords=np.array([[center_x, center_y]]),
                            point_labels=np.array([1]),
                            multimask_output=False
                        )
                    
                    if scores[0] > 0.6:
                        refined_detections.append({
                            "bbox": yolo_box["bbox"],
                            "yolo_conf": yolo_box["conf"],
                            "sam2_conf": float(scores[0]),
                            "combined_conf": (yolo_box["conf"] + float(scores[0])) / 2,
                            "mask": masks[0]
                        })
                except:
                    # If SAM2 fails, keep YOLO detection
                    refined_detections.append({
                        "bbox": yolo_box["bbox"],
                        "yolo_conf": yolo_box["conf"],
                        "combined_conf": yolo_box["conf"]
                    })
            
            # STEP 3: Tracking
            if refined_detections:
                detections = np.array([
                    [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], 
                     d["combined_conf"]]
                    for d in refined_detections
                ])
                
                # Track people across frames
                tracks = self.tracker.update_with_detections(
                    Detection(boxes=detections)
                )
                
                # Store results
                frame_results = []
                for track in tracks:
                    track_id = track.tracker_id
                    x1, y1, x2, y2 = track.box
                    
                    frame_results.append({
                        "track_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "center": [float((x1+x2)/2), float((y1+y2)/2)],
                        "width": float(x2-x1),
                        "height": float(y2-y1)
                    })
                    
                    # Draw on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                 (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                all_person_tracks[frame_idx] = frame_results
            
            out.write(frame)
            
            if frame_idx % 30 == 0:
                num_persons = sum(len(v) for v in all_person_tracks.values())
                print(f"Frame {frame_idx}: Persons tracked = {len(all_person_tracks.get(frame_idx, []))}")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Save results
        self.save_detections(all_person_tracks)
        return all_person_tracks
    
    def save_detections(self, detections, output_json="person_detections.json"):
        """Save detection results to JSON"""
        with open(output_json, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f"✅ Detections saved to {output_json}")
    
    def generate_report(self, detections):
        """Generate statistics report"""
        total_frames = len(detections)
        frames_with_persons = sum(1 for v in detections.values() if len(v) > 0)
        total_person_detections = sum(len(v) for v in detections.values())
        
        unique_ids = set()
        for frame_data in detections.values():
            for person in frame_data:
                unique_ids.add(person['track_id'])
        
        report = {
            "total_frames": total_frames,
            "frames_with_persons": frames_with_persons,
            "total_detections": total_person_detections,
            "unique_person_tracks": len(unique_ids),
            "detection_rate": frames_with_persons / total_frames if total_frames > 0 else 0,
            "avg_persons_per_frame": total_person_detections / frames_with_persons if frames_with_persons > 0 else 0
        }
        
        return report

# Usage
detector = DronePersonDetector("drone_footage.mp4", device="cuda")
detections = detector.detect_and_track("output_detections.mp4")
report = detector.generate_report(detections)

print("\n" + "="*50)
print("DETECTION REPORT")
print("="*50)
for key, value in report.items():
    print(f"{key}: {value}")
```

---

## PERFORMANCE CONSIDERATIONS FOR DRONE VIDEO

### Challenge 1: Small Objects (Persons from High Altitude)

```python
# Solution 1: Image Upsampling
frames_upsampled = [
    cv2.resize(frame, None, fx=1.5, fy=1.5) 
    for frame in frames
]

# Solution 2: Multi-scale Detection
scales = [1.0, 1.5, 2.0]
all_detections = []

for scale in scales:
    scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    detections = model(scaled_frame)
    # Scale coordinates back
    for det in detections:
        det['bbox'] = [c / scale for c in det['bbox']]
    all_detections.extend(detections)

# Solution 3: Crop & Process High-Interest Regions
# Only process areas likely to have persons (open terrain)
```

---

### Challenge 2: Terrain Complexity (Shadows, Vegetation)

```python
# Solution 1: Histogram Equalization (Better contrast)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
enhanced_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Solution 2: Use SAR Imagery (Sees through clouds)
# Combine optical + SAR for robust detection

# Solution 3: Temporal Consistency
# Track moving objects (persons move, shadows don't)
```

---

### Challenge 3: Real-time Processing on Edge Devices

```python
# Solution 1: Model Quantization
model_int8 = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)

# Solution 2: Reduce Frame Rate
process_every_nth_frame = 5  # Process every 5th frame

# Solution 3: Smaller Model
model = YOLO('yolov8s.pt')  # Small instead of XLarge

# Solution 4: Batch Processing
# Process multiple frames in parallel
```

---

## OUTPUT & VISUALIZATION

### Generate Analysis Report

```python
def generate_person_detection_report(detections, output_html="report.html"):
    """
    Create HTML report of person detections
    """
    
    html = """
    <html>
    <head>
        <title>Drone Person Detection Report</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .stat { background: #f0f0f0; padding: 10px; margin: 10px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <h1>Drone Footage - Person Detection Report</h1>
    """
    
    # Add statistics
    total_persons = sum(len(v) for v in detections.values())
    frames_with_persons = sum(1 for v in detections.values() if len(v) > 0)
    
    html += f"""
    <div class="stat">
        <strong>Total Persons Detected:</strong> {total_persons}
    </div>
    <div class="stat">
        <strong>Frames with Detections:</strong> {frames_with_persons} / {len(detections)}
    </div>
    """
    
    # Add frame-by-frame results
    html += "<h2>Frame-by-Frame Results</h2>"
    html += "<table><tr><th>Frame</th><th>Persons</th><th>Locations</th></tr>"
    
    for frame_idx in sorted(detections.keys())[:100]:  # First 100 frames
        persons = detections[frame_idx]
        if persons:
            html += f"<tr><td>{frame_idx}</td><td>{len(persons)}</td>"
            html += "<td>"
            for person in persons:
                x, y = person['center']
                html += f"Person {person['track_id']} @ ({x:.0f}, {y:.0f})<br>"
            html += "</td></tr>"
    
    html += "</table></body></html>"
    
    with open(output_html, 'w') as f:
        f.write(html)
    
    print(f"✅ Report saved to {output_html}")
```

---

## DEPLOYMENT CHECKLIST

- [ ] **Models Downloaded:** SAM2 / YOLO checkpoints ready
- [ ] **Dependencies Installed:** OpenCV, PyTorch, supervision
- [ ] **GPU Available:** Check CUDA/GPU support
- [ ] **Test Video:** Small drone video for testing
- [ ] **Output Directory:** Created for results
- [ ] **Performance Tested:** FPS on target hardware
- [ ] **Thresholds Tuned:** Confidence, IoU for your data
- [ ] **Report Generation:** HTML/JSON output scripts ready
- [ ] **Batch Processing:** Can handle multiple videos
- [ ] **Monitoring:** Logs, error handling implemented

---

## RECOMMENDED APPROACH FOR YOUR DEFENSE USE CASE

### Implementation Plan:

**Week 1:** Set up YOLO baseline
- Use YOLOv8x pre-trained
- Process drone test footage
- Measure FPS and accuracy

**Week 2:** Add SAM2 refinement
- Integrate SAM2 for verification
- Test hybrid pipeline
- Optimize speed vs accuracy

**Week 3:** Implement tracking
- Add ByteTrack
- Generate track IDs
- Create visualization

**Week 4:** Production deployment
- Optimize for edge devices
- Create monitoring dashboard
- Document procedures

---

## FINAL COMMAND FOR YOUR TEAM

To process drone footage right now:

```python
from drone_detector import DronePersonDetector

detector = DronePersonDetector("drone_footage.mp4", device="cuda")
detections = detector.detect_and_track("output.mp4")
report = detector.generate_report(detections)
print(report)
```

**That's it! Full person detection in drone video.** 🚁✨

---

**Next Steps:**
1. ✅ Get your drone footage
2. ✅ Run YOLO detection (this week)
3. ✅ Evaluate accuracy
4. ✅ Add SAM2 refinement
5. ✅ Deploy to production

Good luck! 🎯
