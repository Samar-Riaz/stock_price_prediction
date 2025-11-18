# Voice Classification System - Gate Control

A machine learning system that classifies voices/sounds into "Human/Animal" or "Other" categories to control gate access.

> 📄 **For detailed documentation, see [PROJECT_REPORT.md](PROJECT_REPORT.md)**

## Features

- **Binary Classification**: Distinguishes between Human/Animal voices (gate opens) and Other sounds (gate stays closed)
- **Real-time Recognition**: Live microphone input processing
- **Comprehensive Feature Extraction**: Uses MFCC, Chroma, Spectral Contrast, and Zero Crossing Rate
- **Deep Neural Network**: Trained model with 90%+ accuracy

## Project Structure

```
MLStock/
├── model_training.py          # Train the voice classifier model
├── model_testing.py           # Test model with sample audio files
├── test_all_humans.py         # Test all human voice files
├── test_all_animals.py        # Test all animal files (auto-deletes incorrect)
├── test_all_weapons.py        # Test all weapon files (auto-deletes incorrect)
├── real_time_voice_recognition.py  # Real-time microphone voice recognition
├── voice_classifier_model.h5  # Trained model file
├── feature_scaler.pkl         # Feature scaler for preprocessing
└── dataset/
    ├── human/                 # Human voice samples
    ├── animal/               # Animal sound samples
    └── weapon/                # Other sounds (weapons, etc.)
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For Windows users (sounddevice dependency):**
   ```bash
   pip install sounddevice
   ```
   If you encounter issues, you may need to install PortAudio:
   - Download from: http://files.portaudio.com/download.html
   - Or use: `pip install sounddevice --no-binary sounddevice`

## Usage

### 1. Train the Model

Train a new model with your dataset:
```bash
python model_training.py
```

This will:
- Load audio files from `dataset/human`, `dataset/animal`, and `dataset/weapon`
- Extract comprehensive features
- Train a deep neural network
- Save the model as `voice_classifier_model.h5`
- Save the feature scaler as `feature_scaler.pkl`

### 2. Test the Model

Test the model with sample files:
```bash
python model_testing.py
```

### 3. Real-time Voice Recognition

Start real-time voice recognition from microphone:
```bash
python real_time_voice_recognition.py
```

**Features:**
- Continuously listens to microphone input
- Processes audio in 2-second chunks
- Classifies voices in real-time
- Automatically opens/closes gate based on detection
- Press `Ctrl+C` to stop

**Configuration (in `real_time_voice_recognition.py`):**
- `SAMPLE_RATE`: Audio sample rate (default: 22050 Hz)
- `CHUNK_DURATION`: Seconds of audio per analysis (default: 2.0)
- `THRESHOLD`: Classification threshold (default: 0.5)
- `MIN_VOLUME_THRESHOLD`: Minimum volume to process (default: 0.001)
- `DEBUG_MODE`: Enable debug output (default: True)

### 4. Test All Files

Test all files in a category:
```bash
# Test all human files
python test_all_humans.py

# Test all animal files 
python test_all_animals.py

# Test all weapon files 
python test_all_weapons.py
```

## Model Architecture

The model uses a deep neural network with:
- **Input**: 72 features (MFCC stats, Chroma, Spectral Contrast, ZCR)
- **Architecture**: 
  - Dense(512) → BatchNorm → Dropout(0.4)
  - Dense(256) → BatchNorm → Dropout(0.4)
  - Dense(128) → BatchNorm → Dropout(0.3)
  - Dense(64) → BatchNorm → Dropout(0.3)
  - Dense(32) → Dropout(0.2)
  - Dense(1) → Sigmoid (binary classification)
- **Output**: 0 = Human/Animal (gate opens), 1 = Other (gate closed)

## Gate Control Logic

- **Human/Animal detected** → Gate **OPENS**
- **Other/Unknown sound** → Gate **CLOSES**

## Customization

### Adding Gate Control Hardware

To integrate with actual hardware, modify the `control_gate()` function in `real_time_voice_recognition.py`:

```python
def control_gate(predicted_class, confidence):
    global gate_open
    
    with gate_lock:
        if predicted_class == "human_animal":
            if not gate_open:
                gate_open = True
                # Add your hardware control code here
                # Example: GPIO control, API call, serial communication, etc.
                # gpio.write(GATE_PIN, HIGH)
        else:
            if gate_open:
                gate_open = False
                # Add your hardware control code here
                # gpio.write(GATE_PIN, LOW)
```

### Adjusting Sensitivity

Modify the `THRESHOLD` value in `real_time_voice_recognition.py`:
- Lower threshold (e.g., 0.3) = More sensitive, opens gate more easily
- Higher threshold (e.g., 0.7) = Less sensitive, requires higher confidence

## Troubleshooting

### Microphone Not Working
- Check if microphone is connected and enabled
- Verify audio device selection in the script output
- On Windows, check microphone permissions in Settings

### Low Accuracy
- Retrain the model with more diverse data
- Check audio quality and volume
- Adjust `MIN_VOLUME_THRESHOLD` if needed

### Model Not Found
- Ensure `voice_classifier_model.h5` and `feature_scaler.pkl` exist
- Run `model_training.py` first to generate these files

## Documentation

- **README.md**: Quick start guide and usage instructions
- **PROJECT_REPORT.md**: Comprehensive project documentation including:
  - System architecture and design
  - Technical specifications
  - Performance analysis
  - Use cases and applications
  - Future enhancements

## License

This project is for educational and personal use.

