import pyttsx3
import threading

class AudioFeedback:
    def __init__(self):
        """Initialize text-to-speech engine"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)
        self.is_speaking = False
        
    def speak(self, text):
        """Speak text in a separate thread"""
        if not self.is_speaking:
            self.is_speaking = True
            thread = threading.Thread(target=self._speak_thread, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _speak_thread(self, text):
        """Internal method to speak in thread"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        finally:
            self.is_speaking = False
    
    def announce_detections(self, detections):
        """Announce detected objects"""
        if not detections:
            return
        
        # Group by distance
        close_objects = [d['label'] for d in detections if d['distance'] == 'close']
        medium_objects = [d['label'] for d in detections if d['distance'] == 'medium']
        far_objects = [d['label'] for d in detections if d['distance'] == 'far']
        
        announcement = []
        
        if close_objects:
            announcement.append(f"{len(close_objects)} objects nearby: {', '.join(set(close_objects))}")
        if medium_objects:
            announcement.append(f"{len(medium_objects)} objects at medium distance")
        if far_objects:
            announcement.append(f"{len(far_objects)} objects far away")
        
        if announcement:
            self.speak(". ".join(announcement))
