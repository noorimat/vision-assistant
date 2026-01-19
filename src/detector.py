import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize YOLO model"""
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
        
    def detect_objects(self, frame):
        """Run detection on a frame"""
        results = self.model(frame, verbose=False)
        return results[0]
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        boxes = results.boxes
        
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            
            # Only show detections with confidence > 0.5
            if confidence > 0.5:
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def get_detections_list(self, results, confidence_threshold=0.5):
        """Get list of detected objects with their info"""
        detections = []
        boxes = results.boxes
        
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence > confidence_threshold:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                
                # Calculate distance (simple approximation based on box size)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                # Larger box = closer object
                distance = "close" if box_area > 50000 else "medium" if box_area > 20000 else "far"
                
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'distance': distance
                })
        
        return detections
