import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_feedback import AudioFeedback

class TestAudioFeedback:
    
    @patch('audio_feedback.pyttsx3.init')
    def test_audio_feedback_initialization(self, mock_init):
        """Test that AudioFeedback initializes properly"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        assert audio.engine is not None
        assert mock_engine.setProperty.call_count == 2  # rate and volume
        assert not audio.is_speaking
    
    @patch('audio_feedback.pyttsx3.init')
    def test_speak_method_exists(self, mock_init):
        """Test that speak method works"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        # Should not raise exception
        audio.speak("test message")
        
        # Give thread a moment
        time.sleep(0.2)
    
    @patch('audio_feedback.pyttsx3.init')
    def test_announce_detections_with_close_objects(self, mock_init):
        """Test announcement for close objects"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        # Mock the speak method to track calls
        with patch.object(audio, 'speak') as mock_speak:
            detections = [
                {'label': 'person', 'confidence': 0.9, 'distance': 'close'},
                {'label': 'chair', 'confidence': 0.8, 'distance': 'close'},
            ]
            
            audio.announce_detections(detections)
            
            # Verify speak was called
            mock_speak.assert_called_once()
            call_args = mock_speak.call_args[0][0]
            assert 'nearby' in call_args.lower() or 'close' in call_args.lower()
            assert '2' in call_args
    
    @patch('audio_feedback.pyttsx3.init')
    def test_announce_detections_with_mixed_distances(self, mock_init):
        """Test announcement with objects at different distances"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        with patch.object(audio, 'speak') as mock_speak:
            detections = [
                {'label': 'person', 'confidence': 0.9, 'distance': 'close'},
                {'label': 'car', 'confidence': 0.8, 'distance': 'medium'},
                {'label': 'dog', 'confidence': 0.7, 'distance': 'far'},
            ]
            
            audio.announce_detections(detections)
            
            mock_speak.assert_called_once()
            call_args = mock_speak.call_args[0][0]
            
            # Should mention different distance categories
            assert 'nearby' in call_args.lower() or 'close' in call_args.lower()
    
    @patch('audio_feedback.pyttsx3.init')
    def test_announce_detections_empty_list(self, mock_init):
        """Test that empty detections don't trigger announcement"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        with patch.object(audio, 'speak') as mock_speak:
            audio.announce_detections([])
            
            # Should not call speak for empty list
            mock_speak.assert_not_called()
    
    @patch('audio_feedback.pyttsx3.init')
    def test_announce_detections_groups_duplicate_objects(self, mock_init):
        """Test that duplicate objects are grouped properly"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        with patch.object(audio, 'speak') as mock_speak:
            detections = [
                {'label': 'person', 'confidence': 0.9, 'distance': 'close'},
                {'label': 'person', 'confidence': 0.85, 'distance': 'close'},
                {'label': 'person', 'confidence': 0.8, 'distance': 'close'},
            ]
            
            audio.announce_detections(detections)
            
            mock_speak.assert_called_once()
            call_args = mock_speak.call_args[0][0]
            
            # Should say "3 objects"
            assert '3' in call_args
    
    @patch('audio_feedback.pyttsx3.init')
    def test_announce_detections_medium_distance(self, mock_init):
        """Test announcement for medium distance objects"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        with patch.object(audio, 'speak') as mock_speak:
            detections = [
                {'label': 'car', 'confidence': 0.8, 'distance': 'medium'},
            ]
            
            audio.announce_detections(detections)
            
            mock_speak.assert_called_once()
            call_args = mock_speak.call_args[0][0]
            assert 'medium' in call_args.lower()
    
    @patch('audio_feedback.pyttsx3.init')
    def test_announce_detections_far_distance(self, mock_init):
        """Test announcement for far distance objects"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        with patch.object(audio, 'speak') as mock_speak:
            detections = [
                {'label': 'bird', 'confidence': 0.7, 'distance': 'far'},
            ]
            
            audio.announce_detections(detections)
            
            mock_speak.assert_called_once()
            call_args = mock_speak.call_args[0][0]
            assert 'far' in call_args.lower()
    
    @patch('audio_feedback.pyttsx3.init')
    def test_is_speaking_flag(self, mock_init):
        """Test that is_speaking flag prevents overlapping speech"""
        mock_engine = Mock()
        mock_init.return_value = mock_engine
        
        audio = AudioFeedback()
        
        # Initially not speaking
        assert not audio.is_speaking
        
        # Simulate speaking
        audio.is_speaking = True
        
        # Try to speak while already speaking - should be blocked
        audio.speak("test")
        
        # Flag should still be true
        assert audio.is_speaking
