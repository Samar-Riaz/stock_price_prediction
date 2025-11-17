import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight  # Added for class weights

# Function to extract comprehensive audio features
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
                            spectral_contrast = np.zeros((7, 1))  # 7 bands, 1 frame
            else:
                # For other ValueError, use zeros
                spectral_contrast = np.zeros((7, 1))
        except Exception:
            # Any other exception - use zeros
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
        # Silently return zeros for failed extractions (don't print to avoid spam)
        # Return zeros if extraction fails
        return np.zeros(13 * 4 + 12 + 7 + 1)  # Total feature size

# Load data (from human, animal, and other/unknown folders)
def load_data():
    human_dir = 'dataset/human'
    animal_dir = 'dataset/animal'
    other_dir = 'dataset/test'  # Other/unknown sounds

    X = []  # Features (MFCCs)
    y = []  # Labels: 'human_animal' (0) or 'other' (1)

    # Load human audio data - label as 'human_animal' (class 0)
    print("Loading human audio files...")
    human_count = 0
    for filename in os.listdir(human_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(human_dir, filename)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append('human_animal')  # Combined class for human and animal
                human_count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    print(f"Loaded {human_count} human audio files")

    # Load animal audio data - label as 'human_animal' (class 0)
    print("Loading animal audio files...")
    animal_count = 0
    for filename in os.listdir(animal_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(animal_dir, filename)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append('human_animal')  # Combined class for human and animal
                animal_count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    print(f"Loaded {animal_count} animal audio files")

    # Load other/unknown audio data - label as 'other' (class 1)
    # Include both test folder and weapon folder as "other" class
    other_dirs = ['dataset/test', 'dataset/weapon']
    print("Loading other/unknown audio files...")
    other_count = 0
    for other_dir in other_dirs:
        if os.path.exists(other_dir):
            for filename in os.listdir(other_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(other_dir, filename)
                    try:
                        features = extract_features(file_path)
                        X.append(features)
                        y.append('other')  # Other/unknown sounds
                        other_count += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
    print(f"Loaded {other_count} other/unknown audio files")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"\nTotal samples: {len(X)}")
    print(f"  - Human/Animal (class 0): {np.sum(y == 'human_animal')}")
    print(f"  - Other/Unknown (class 1): {np.sum(y == 'other')}")

    return X, y

# Step 2: Model Selection and Training
def train_model(X, y):
    # Encode labels: 'human_animal' -> 0, 'other' -> 1
    # This is binary classification: 0 = human/animal (gate opens), 1 = other (gate closed)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Verify encoding
    print(f"\nLabel encoding:")
    for label in label_encoder.classes_:
        encoded = label_encoder.transform([label])[0]
        print(f"  {label} -> {encoded}")

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Normalize the data using StandardScaler (better than simple max normalization)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler for later use
    import pickle
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Feature scaler saved to 'feature_scaler.pkl'")

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    # Build a deeper and more sophisticated neural network model
    # Binary classification: 0 = human/animal, 1 = other
    model = tf.keras.Sequential([
        # Input layer with L2 regularization
        tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],),
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # First hidden layer
        tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Second hidden layer
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Third hidden layer
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Fourth hidden layer
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer (binary classification)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with optimized settings
    # Use a learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Add callbacks for better training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # Monitor accuracy instead of loss
        patience=15,  # More patience
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,  # More patience before reducing LR
        min_lr=0.000001,
        verbose=1
    )
    
    # Model checkpoint to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )

    print("\nTraining model...")
    print(f"Input feature size: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Model parameters: {model.count_params():,}")
    
    # Train the model with class weights and callbacks
    history = model.fit(
        X_train, y_train, 
        epochs=100,  # More epochs
        batch_size=128,  # Larger batch size for better generalization
        validation_data=(X_test, y_test), 
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    # Evaluate the model
    print("\nEvaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test precision: {test_results[2]:.4f}")
    print(f"Test recall: {test_results[3]:.4f}")
    
    # Calculate F1 score
    f1_score = 2 * (test_results[2] * test_results[3]) / (test_results[2] + test_results[3] + 1e-7)
    print(f"Test F1 score: {f1_score:.4f}")
    
    # Predict on test set to get confusion matrix
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['human_animal', 'other']))

    # Save the final model
    model.save("voice_classifier_model.h5")
    print("\nModel saved as 'voice_classifier_model.h5'")
    
    # If best model was saved, inform user
    import os
    if os.path.exists('best_model.h5'):
        print("Best model (based on validation accuracy) saved as 'best_model.h5'")
        print("You can use 'best_model.h5' if it performs better than 'voice_classifier_model.h5'")
    
    return model

# Main Program
if __name__ == "__main__":
    # Load data
    X, y = load_data()

    # Train the model
    train_model(X, y)
