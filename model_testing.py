import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]

# Function to extract comprehensive audio features (same as training)
def extract_features(audio_path, n_mfcc=13):
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
        try:
            # Try with default parameters first
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        except ValueError as e:
            if "Nyquist" in str(e) or "frequency band" in str(e).lower():
                # Reduce number of bands if Nyquist error
                try:
                    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=3)
                except:
                    # Last resort: use fewer bands
                    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=2)
            else:
                raise
        
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
        print(f"Error extracting features from {audio_path}: {e}")
        # Return zeros if extraction fails
        return np.zeros(13 * 4 + 12 + 7 + 1)  # Total feature size

# Load the trained model and scaler
print("Loading model and scaler...")
model = load_model("voice_classifier_model.h5")
try:
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("Warning: Scaler file not found. Using model without scaling.")
    scaler = None

# Function to classify the voice
# New model: 0 = human/animal (gate opens), 1 = other (gate closed)
def classify_voice(audio_path, threshold=0.5):
    """
    Classify voice using the new model structure.
    
    Args:
        audio_path: Path to audio file
        threshold: Decision threshold (default: 0.5)
                   < threshold = human/animal (gate opens)
                   >= threshold = other (gate closed)
    
    Returns:
        predicted_class: "human_animal" or "other"
        confidence: The raw prediction value (0=human/animal, 1=other)
    """
    # Extract features using the same method as training
    features = extract_features(audio_path)
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

# Test different audio files
def test_audio_file(test_audio_file):
    predicted_class, confidence = classify_voice(test_audio_file)
    
    # Display prediction with confidence
    # confidence is the probability of "other" class
    # So (1 - confidence) is the probability of "human_animal" class
    human_animal_prob = (1 - confidence) * 100
    other_prob = confidence * 100
    
    print(f"Testing {test_audio_file}:")
    print(f"  Predicted: {predicted_class.upper()}")
    print(f"  Raw confidence: {confidence:.4f} ({human_animal_prob:.1f}% human/animal, {other_prob:.1f}% other)")

    # Gate opening logic
    if predicted_class == "human_animal":
        print(f"  ✓ Human/Animal voice detected. Opening the gate...")
        print("  → Gate Opened!")
    else:
        print(f"  ✗ Unknown/Other sound detected. Gate remains closed.")
        print("  → Gate Closed!")
    print()  # Empty line for readability

# Example test files - Update these paths as needed
test_audio_files = [
    "dataset/human/0_george_1.wav",  # Example human file
    "dataset/animal/Animal Ambience Monkeys In Jungle 01.wav",  # Example animal file
    "dataset/weapon/Weapon Axe Drop 01.wav"  # Example other sound
]

# Test each file
for file in test_audio_files:
    test_audio_file(file)
