import cv2
import time
from detector import ObjectDetector
from audio_feedback import AudioFeedback

def main():
    print("Vision Assistant - Real-Time Object Detection")
    print("=" * 50)
    print("Controls:")
    print("  Q - Quit")
    print("  S - Toggle sound")
    print("=" * 50)
    
    # Initialize components
    detector = ObjectDetector('yolov8n.pt')
    audio = AudioFeedback()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    sound_enabled = True
    last_announcement = time.time()
    announcement_interval = 3  # Announce every 3 seconds
    
    print("\nStarting detection... Press 'Q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = detector.detect_objects(frame)
        
        # Draw detections
        frame = detector.draw_detections(frame, results)
        
        # Audio feedback (every 3 seconds)
        current_time = time.time()
        if sound_enabled and (current_time - last_announcement) > announcement_interval:
            detections = detector.get_detections_list(results)
            if detections:
                audio.announce_detections(detections)
            last_announcement = current_time
        
        # Add status text
        status = "Sound: ON" if sound_enabled else "Sound: OFF"
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Vision Assistant', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            sound_enabled = not sound_enabled
            print(f"Sound {'enabled' if sound_enabled else 'disabled'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nVision Assistant stopped.")

if __name__ == "__main__":  # pragma: no cover
    main()
