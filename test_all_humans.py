import numpy as np
import librosa
import pickle
import os
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]

# Function to extract comprehensive audio features (same as training)
def extract_features(audio_path, n_mfcc=13, debug=False):
    """
    Extract comprehensive audio features including:
    - MFCCs (mean, std, min, max)
    - Chroma features
    - Spectral contrast
    - Zero crossing rate
    """
    try:
        # Load audio file with librosa
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Pad audio if too short (minimum 2048 samples for n_fft=2048)
        min_length = 2048
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        # Extract MFCCs (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Extract additional features with error handling
        # Chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=min(2048, len(audio)))
        except:
            # Fallback for very short files
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=512)
        
        # Spectral contrast with error handling
        spectral_contrast = None
        try:
            # Try with default parameters first
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        except ValueError as e:
            if "Nyquist" in str(e) or "frequency band" in str(e).lower():
                # Reduce number of bands if Nyquist error
                try:
                    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=3)
                except:
                    try:
                        # Try with even fewer bands
                        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=2)
                    except:
                        try:
                            # Last resort: try with n_bands=1
                            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=1)
                        except:
                            # If all fails, create dummy spectral contrast (7 bands of zeros)
                            if debug:
                                print(f"  Warning: Spectral contrast failed for {audio_path}, using zeros")
                            spectral_contrast = np.zeros((7, 1))  # 7 bands, 1 frame
            else:
                # For other ValueError, try to create dummy
                if debug:
                    print(f"  Warning: Spectral contrast ValueError for {audio_path}, using zeros")
                spectral_contrast = np.zeros((7, 1))
        except Exception as e:
            # Any other exception - use zeros
            if debug:
                print(f"  Warning: Spectral contrast exception for {audio_path}: {e}, using zeros")
            spectral_contrast = np.zeros((7, 1))
        
        zcr = librosa.feature.zero_crossing_rate(audio)
        
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
        if debug:
            print(f"Error in extract_features for {audio_path}: {e}")
        return None

# Load the trained model and scaler
print("Loading model and scaler...")
model = load_model("voice_classifier_model.h5")
try:
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully!\n")
except FileNotFoundError:
    print("Warning: Scaler file not found. Using model without scaling.")
    scaler = None

# Function to classify the voice
def classify_voice(audio_path, threshold=0.5):
    """
    Classify voice using the model structure.
    Model output: 0 = human/animal (gate opens), 1 = other (gate closed)
    """
    try:
        # Extract features using the same method as training
        features = extract_features(audio_path)
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
        # Return None on error - let caller handle it
        return None, None

# Get all human files
human_dir = 'dataset/human'
all_files = [f for f in os.listdir(human_dir) if f.endswith('.wav')]

print("=" * 80)
print("TESTING ALL HUMAN FILES")
print("=" * 80)
print(f"Total human files found: {len(all_files)}\n")

# Test each file
correct_files = []
incorrect_files = []
error_files = []
file_results = []

# Test all files
files_to_test = all_files

print(f"Testing all {len(files_to_test)} human files...\n")

for i, filename in enumerate(files_to_test, 1):
    file_path = os.path.join(human_dir, filename)
    
    if (i % 100 == 0) or i == 1:
        print(f"Progress: [{i}/{len(files_to_test)}] Testing files...")
    
    try:
        predicted_class, confidence = classify_voice(file_path)
        
        if predicted_class is None:
            # Try to extract features to see what the error is
            try:
                features = extract_features(file_path, debug=(len(error_files) < 3))
                if features is None:
                    error_files.append((filename, "Feature extraction returned None"))
                else:
                    # Features extracted but classification failed - check if it's a shape issue
                    try:
                        features_reshaped = features.reshape(1, -1)
                        if scaler is not None:
                            features_scaled = scaler.transform(features_reshaped)
                        else:
                            features_scaled = features_reshaped
                        error_files.append((filename, f"Classification failed (features shape: {features.shape}, scaled shape: {features_scaled.shape})"))
                    except Exception as e2:
                        error_files.append((filename, f"Scaling/reshaping error: {str(e2)[:80]}"))
            except Exception as e:
                error_files.append((filename, f"Feature extraction exception: {str(e)[:100]}"))
            continue
        
        # Determine if correct (should be human_animal)
        is_correct = predicted_class == "human_animal"
        
        result = {
            'filename': filename,
            'predicted': predicted_class,
            'confidence': confidence,
            'correct': is_correct
        }
        file_results.append(result)
        
        if is_correct:
            correct_files.append((filename, confidence))
        else:
            incorrect_files.append((filename, confidence))
            
    except Exception as e:
        error_files.append((filename, str(e)))
        if len(error_files) <= 5:  # Show first 5 errors
            print(f"  Error with {filename}: {e}")
        continue

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total files tested: {len(files_to_test)}")
print(f"✓ Correct (detected as human_animal): {len(correct_files)}")
print(f"✗ Incorrect (detected as other): {len(incorrect_files)}")
print(f"⚠ Errors: {len(error_files)}")
if len(files_to_test) > 0:
    accuracy = (len(correct_files) / len(files_to_test)) * 100
    print(f"📊 Overall Accuracy: {accuracy:.1f}%")
    
    # Calculate average confidence
    if correct_files:
        avg_correct_conf = np.mean([x[1] for x in correct_files])
        print(f"📊 Average confidence (correct): {avg_correct_conf:.4f}")
    if incorrect_files:
        avg_incorrect_conf = np.mean([x[1] for x in incorrect_files])
        print(f"📊 Average confidence (incorrect): {avg_incorrect_conf:.4f}")
print("=" * 80)

# Confidence distribution analysis
if file_results:
    print("\n" + "=" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)
    
    correct_confidences = [r['confidence'] for r in file_results if r['correct']]
    incorrect_confidences = [r['confidence'] for r in file_results if not r['correct']]
    
    if correct_confidences:
        print(f"\nCorrect detections (human_animal):")
        print(f"  Min confidence: {min(correct_confidences):.4f}")
        print(f"  Max confidence: {max(correct_confidences):.4f}")
        print(f"  Mean confidence: {np.mean(correct_confidences):.4f}")
        print(f"  Median confidence: {np.median(correct_confidences):.4f}")
    
    if incorrect_confidences:
        print(f"\nIncorrect detections (other):")
        print(f"  Min confidence: {min(incorrect_confidences):.4f}")
        print(f"  Max confidence: {max(incorrect_confidences):.4f}")
        print(f"  Mean confidence: {np.mean(incorrect_confidences):.4f}")
        print(f"  Median confidence: {np.median(incorrect_confidences):.4f}")

# Show some examples of incorrect classifications
if incorrect_files:
    print("\n" + "=" * 80)
    print("SAMPLE OF INCORRECTLY CLASSIFIED FILES (First 20)")
    print("=" * 80)
    for filename, confidence in incorrect_files[:20]:
        print(f"  ✗ {filename[:60]:60s} | Confidence: {confidence:.4f}")

# Save detailed results to file
with open('human_test_report.txt', 'w', encoding='utf-8') as f:
    f.write("HUMAN CLASSIFICATION TEST REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total files tested: {len(files_to_test)}\n")
    f.write(f"Correct (human_animal): {len(correct_files)}\n")
    f.write(f"Incorrect (other): {len(incorrect_files)}\n")
    f.write(f"Errors: {len(error_files)}\n")
    if len(files_to_test) > 0:
        f.write(f"Overall Accuracy: {(len(correct_files)/len(files_to_test)*100):.1f}%\n\n")
    
    if correct_files:
        avg_correct_conf = np.mean([x[1] for x in correct_files])
        f.write(f"Average confidence (correct): {avg_correct_conf:.4f}\n")
    if incorrect_files:
        avg_incorrect_conf = np.mean([x[1] for x in incorrect_files])
        f.write(f"Average confidence (incorrect): {avg_incorrect_conf:.4f}\n\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("DETAILED RESULTS BY FILE\n")
    f.write("=" * 80 + "\n\n")
    
    # Sort by filename
    file_results_sorted = sorted(file_results, key=lambda x: x['filename'])
    
    for result in file_results_sorted:
        status = "[OK]" if result['correct'] else "[X]"
        f.write(f"{status} {result['filename']:70s} | "
                f"Predicted: {result['predicted']:15s} | Confidence: {result['confidence']:.4f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("INCORRECTLY CLASSIFIED FILES\n")
    f.write("=" * 80 + "\n\n")
    for filename, confidence in sorted(incorrect_files):
        f.write(f"[X] {filename:70s} | Confidence: {confidence:.4f}\n")
    
    if error_files:
        f.write("\n" + "=" * 80 + "\n")
        f.write("FILES WITH ERRORS\n")
        f.write("=" * 80 + "\n\n")
        for error_item in sorted(error_files):
            if isinstance(error_item, tuple):
                filename, error_msg = error_item
                f.write(f"[ERROR] {filename:70s} | {error_msg}\n")
            else:
                f.write(f"[ERROR] {error_item}\n")

print(f"\n✓ Detailed report saved to 'human_test_report.txt'")
print("=" * 80)

