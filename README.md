# Vision Assistant - Real-Time Object Detection with Audio Feedback

Real-time object detection system using YOLOv8 and OpenCV with text-to-speech announcements. Detects objects through your webcam and provides audio feedback about their proximity.

[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)]()

## Features

- **Real-time detection** using YOLOv8n (lightweight YOLO model)
- **Audio announcements** via text-to-speech for detected objects
- **Distance estimation** based on bounding box size (close/medium/far)
- **Visual feedback** with bounding boxes and confidence scores
- **Toggleable audio** for flexible use
- **100% test coverage** with comprehensive unit and integration tests

## Demo

The system detects objects in real-time and announces them every 3 seconds:
- "2 objects nearby: person, chair"
- "1 object at medium distance"
- Visual bounding boxes with labels and confidence scores

## Requirements

- Python 3.11+
- Webcam
- macOS/Linux/Windows

## Installation

1. Clone the repository:
```bash
git clone https://github.com/noorimat/vision-assistant.git
cd vision-assistant
```

2. Create virtual environment with Python 3.11:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Upgrade pip and install dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install opencv-python numpy ultralytics pyttsx3
```

## Usage

Run the vision assistant:
```bash
python src/main.py
```

### Controls

- **Q** - Quit application
- **S** - Toggle sound on/off

### First Run

The first time you run the application, it will download the YOLOv8n model (~6MB). Subsequent runs will use the cached model.

## Testing

This project includes comprehensive unit tests with **100% code coverage**.

### Running Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests (use python -m pytest, not just pytest)
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage report
python -m pytest --cov=src --cov-report=html

# View coverage report in browser
open htmlcov/index.html
```

### Test Coverage

**🎯 100% Code Coverage Achieved!**

- **audio_feedback.py**: 100% ✅
- **detector.py**: 100% ✅
- **main.py**: 100% ✅

**Total: 33 tests, all passing**

### Test Structure
```
tests/
├── conftest.py                      # Shared fixtures and mocks
├── test_audio_feedback.py           # AudioFeedback unit tests (9 tests)
├── test_detector.py                 # ObjectDetector unit tests (10 tests)
├── test_main.py                     # Main integration tests (10 tests)
├── test_main_complete_coverage.py   # Additional main coverage tests (2 tests)
└── test_main_script.py             # Script execution tests (2 tests)
```

### Tests Included

**Audio Feedback (100% coverage - 9 tests)**
- ✅ Initialization and TTS engine setup
- ✅ Speech threading and concurrency
- ✅ Announcement grouping by distance
- ✅ Empty detection handling
- ✅ Duplicate object grouping
- ✅ Distance-specific announcements

**Object Detector (100% coverage - 10 tests)**
- ✅ YOLO model initialization
- ✅ Detection pipeline execution
- ✅ Confidence threshold filtering
- ✅ Distance estimation (close/medium/far)
- ✅ Bounding box and label drawing
- ✅ Detection data structure validation
- ✅ Visual element rendering

**Main Application (100% coverage - 14 tests)**
- ✅ Module imports and structure
- ✅ Webcam initialization and failure handling
- ✅ Key event handling (quit, sound toggle)
- ✅ Frame read failure recovery
- ✅ Audio announcement timing logic
- ✅ User instruction display
- ✅ Script execution as __main__

## How It Works

1. **Object Detection**: Uses YOLOv8n to detect 80+ common objects in real-time
2. **Distance Estimation**: Calculates approximate distance based on bounding box area
   - Close: Box area > 50,000 pixels
   - Medium: Box area > 20,000 pixels
   - Far: Box area < 20,000 pixels
3. **Audio Feedback**: Announces detected objects every 3 seconds using pyttsx3
4. **Visual Display**: Shows bounding boxes with labels and confidence scores

## Project Structure
```
vision-assistant/
├── src/
│   ├── main.py              # Main application (100% coverage)
│   ├── detector.py          # YOLO detection logic (100% coverage)
│   └── audio_feedback.py    # TTS announcements (100% coverage)
├── tests/                   # 33 comprehensive tests
│   ├── conftest.py
│   ├── test_audio_feedback.py
│   ├── test_detector.py
│   ├── test_main.py
│   ├── test_main_complete_coverage.py
│   └── test_main_script.py
├── models/                  # YOLO models (auto-downloaded)
├── venv/                    # Virtual environment
├── pytest.ini              # Pytest configuration
├── requirements.txt
└── README.md
```

## Technical Details

- **Model**: YOLOv8n (nano variant for fast inference)
- **Detection Threshold**: 50% confidence minimum
- **Announcement Interval**: 3 seconds
- **Supported Objects**: 80 COCO dataset classes (person, car, chair, etc.)
- **Test Framework**: pytest with 100% code coverage

## Troubleshooting

### NumPy Installation Issues
If you encounter NumPy errors on macOS, ensure you're using Python 3.11:
```bash
brew install python@3.11
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv
```

### Webcam Access Denied
Grant camera permissions in System Preferences → Security & Privacy → Camera

### No Audio Output
Check system volume and ensure pyttsx3 has proper permissions

### Pytest Command Not Found
Always use `python -m pytest` instead of just `pytest` to ensure the correct Python environment

## Future Enhancements

- [ ] Add depth camera support for accurate distance measurement
- [ ] Implement custom object training
- [ ] Add gesture recognition
- [ ] Mobile app integration
- [ ] Multi-language support
- [ ] Object tracking across frames
- [ ] Web dashboard for remote monitoring
- [ ] Database logging for detection history

## License

MIT License - see LICENSE file for details

## Author

Built by [@noorimat](https://github.com/noorimat)

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3)
