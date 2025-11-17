import numpy as np
import librosa
import pickle
import sounddevice as sd  # pyright: ignore[reportMissingImports]
import threading
import time
from collections import deque
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]

# Configuration
SAMPLE_RATE = 22050  # librosa default
CHUNK_DURATION = 2.0  # seconds of audio to analyze at a time
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
THRESHOLD = 0.5  # Classification threshold
MIN_VOLUME_THRESHOLD = 0.001  # Minimum volume to consider as valid audio (lowered for better sensitivity)
DEBUG_MODE = True  # Show debug information

# Gate state
gate_open = False
gate_lock = threading.Lock()

# Load the trained model and scaler
print("Loading model and scaler...")
try:
    model = load_model("voice_classifier_model.h5")
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Model and scaler loaded successfully!\n")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Make sure 'voice_classifier_model.h5' and 'feature_scaler.pkl' exist in the current directory.")
    exit(1)

def extract_features_from_audio(audio_data, sr=SAMPLE_RATE, n_mfcc=13):
    """
    Extract comprehensive audio features from audio array (same as training).
    """
    try:
        # Ensure audio is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # Pad audio if too short (minimum 2048 samples for n_fft=2048)
        min_length = 2048
        if len(audio_data) < min_length:
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')
        
        # Extract MFCCs (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Extract additional features with error handling
        # Chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=min(2048, len(audio_data)))
        except:
            # Fallback for very short files
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=512)
        
        # Spectral contrast with error handling
        spectral_contrast = None
        try:
            # Try with default parameters first
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        except ValueError as e:
            if "Nyquist" in str(e) or "frequency band" in str(e).lower():
                # Reduce number of bands if Nyquist error
                try:
                    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_bands=3)
                except:
                    try:
                        # Try with even fewer bands
                        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_bands=2)
                    except:
                        try:
                            # Last resort: try with n_bands=1
                            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_bands=1)
                        except:
                            # If all fails, create dummy spectral contrast (7 bands of zeros)
                            spectral_contrast = np.zeros((7, 1))  # 7 bands, 1 frame
            else:
                # For other ValueError, try to create dummy
                spectral_contrast = np.zeros((7, 1))
        except Exception as e:
            # Any other exception - use zeros
            spectral_contrast = np.zeros((7, 1))
        
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        
        # Compute statistics for each feature
        features = []
        
        # MFCC statistics (mean, std, min, max for each coefficient)
        features.extend(np.mean(mfccs, axis=1))  # Mean
        features.extend(np.std(mfccs, axis=1))    # Standard deviation
        features.extend(np.min(mfccs, axis=1))    # Minimum
        features.extend(np.max(mfccs, axis=1))    # Maximum
        
        # Chroma features (mean of each chroma bin)
        features.extend(np.mean(chroma, axis=1))
        
        # Spectral contrast (mean) - handle variable number of bands
        spectral_mean = np.mean(spectral_contrast, axis=1)
        # Pad or truncate to expected size (7 bands)
        if len(spectral_mean) < 7:
            spectral_mean = np.pad(spectral_mean, (0, 7 - len(spectral_mean)), mode='constant')
        elif len(spectral_mean) > 7:
            spectral_mean = spectral_mean[:7]
        features.extend(spectral_mean)
        
        # Zero crossing rate (mean)
        features.append(np.mean(zcr))
        
        return np.array(features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def classify_voice(audio_data, threshold=THRESHOLD):
    """
    Classify voice using the model.
    Model output: 0 = human/animal (gate opens), 1 = other (gate closed)
    """
    try:
        # Extract features
        features = extract_features_from_audio(audio_data)
        if features is None:
            return None, None
        
        features = features.reshape(1, -1)  # Reshape to match the input shape
        
        # Apply the same scaling as training
        if scaler is not None:
            features = scaler.transform(features)
        
        # Predict the class
        # Model output: 0 = human/animal, 1 = other
        prediction = model.predict(features, verbose=0)
        confidence = float(prediction[0][0])
        
        # Classification logic:
        # If confidence < 0.5, it's closer to 0 (human/animal)
        # If confidence >= 0.5, it's closer to 1 (other)
        if confidence < threshold:
            predicted_class = "human_animal"
        else:
            predicted_class = "other"
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error in classification: {e}")
        return None, None

def control_gate(predicted_class, confidence):
    """
    Control gate based on voice classification.
    """
    global gate_open
    
    with gate_lock:
        if predicted_class == "human_animal":
            if not gate_open:
                gate_open = True
                print(f"🚪 GATE OPENED - Human/Animal detected (confidence: {1-confidence:.3f})")
                # Here you would add actual gate control code
                # e.g., GPIO control, API call, etc.
            return True
        else:  # other
            if gate_open:
                gate_open = False
                print(f"🚪 GATE CLOSED - Unknown/Other sound detected (confidence: {confidence:.3f})")
                # Here you would add actual gate control code
            return False

def process_audio_chunk(audio_chunk):
    """
    Process a chunk of audio and control gate.
    """
    # Check if audio has sufficient volume
    volume = np.abs(audio_chunk).mean()
    max_volume = np.abs(audio_chunk).max()
    
    if DEBUG_MODE:
        # Show volume level every time (helps debug)
        if volume < MIN_VOLUME_THRESHOLD:
            print(f"🔇 Volume too low: {volume:.6f} (min: {MIN_VOLUME_THRESHOLD:.6f}) | Max: {max_volume:.6f}")
            return  # Too quiet, ignore
        else:
            print(f"🎤 Audio detected! Volume: {volume:.6f} | Max: {max_volume:.6f} | Processing...")
    
    if volume < MIN_VOLUME_THRESHOLD:
        return  # Too quiet, ignore
    
    # Classify the audio
    predicted_class, confidence = classify_voice(audio_chunk)
    
    if predicted_class is None:
        if DEBUG_MODE:
            print("⚠️  Classification failed (returned None)")
        return
    
    # Control gate based on classification
    control_gate(predicted_class, confidence)
    
    # Display status
    status = "OPEN" if gate_open else "CLOSED"
    class_display = "Human/Animal" if predicted_class == "human_animal" else "Other/Unknown"
    conf_display = (1 - confidence) if predicted_class == "human_animal" else confidence
    print(f"[{status}] {class_display} | Confidence: {conf_display:.3f} | Volume: {volume:.4f}")

def audio_callback(indata, frames, time_info, status):
    """
    Callback function for sounddevice audio stream.
    """
    if status:
        print(f"⚠️  Audio status: {status}")
    
    # Convert to mono if stereo
    if len(indata.shape) > 1:
        audio_chunk = np.mean(indata, axis=1)
    else:
        audio_chunk = indata.flatten()
    
    # Process audio in a separate thread to avoid blocking
    threading.Thread(target=process_audio_chunk, args=(audio_chunk,), daemon=True).start()

def main():
    """
    Main function to start real-time voice recognition.
    """
    print("=" * 80)
    print("REAL-TIME VOICE RECOGNITION - GATE CONTROL SYSTEM")
    print("=" * 80)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Chunk Duration: {CHUNK_DURATION} seconds")
    print(f"Classification Threshold: {THRESHOLD}")
    print(f"Minimum Volume Threshold: {MIN_VOLUME_THRESHOLD}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("\nListening for voices...")
    print("\n⚠️  TO STOP: Press Ctrl+C (or Ctrl+Break on Windows)")
    print("   The system will continue listening until you stop it manually.")
    if DEBUG_MODE:
        print("\n💡 DEBUG MODE: You'll see volume levels and processing status.")
    print("\n" + "=" * 80)
    
    try:
        # List available audio devices
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"  [{i}] {device['name']}{default}")
        print()
        
        # Start audio stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,  # Mono
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            dtype=np.float32
        ):
            print("✓ Microphone active. Listening...")
            print("   Processing audio every 2 seconds...")
            print("   Speak into your microphone to test!\n")
            try:
                while True:
                    time.sleep(0.1)  # Keep the main thread alive
            except KeyboardInterrupt:
                pass  # Will be handled in outer try-except
                
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Stopping voice recognition...")
        print("=" * 80)
        with gate_lock:
            if gate_open:
                print("🚪 Closing gate...")
                gate_open = False
        print("✓ System stopped.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

