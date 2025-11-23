import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def extract_features(file_path, duration=30):
    """Extract comprehensive audio features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=duration, sr=22050)
        
        # 1. MFCCs (20 coefficients for better representation)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)  # Mean across time
        mfccs_std = np.std(mfccs, axis=1)    # Std across time
        
        # 2. Chroma features (important for musical genre)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # 3. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = float(np.mean(spectral_centroid))
        spectral_centroid_std = float(np.std(spectral_centroid))
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))
        spectral_rolloff_std = float(np.std(spectral_rolloff))
        
        # 5. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))
        spectral_bandwidth_std = float(np.std(spectral_bandwidth))
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        
        # 7. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)  # Ensure it's a scalar
        
        # 8. RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        
        # 9. Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_spec_mean = np.mean(mel_spec, axis=1)  # Mean across time
        mel_spec_std = np.std(mel_spec, axis=1)    # Std across time
        
        # Combine all features into a single 1D array
        features = np.concatenate([
            mfccs_mean,                          # 20 features
            mfccs_std,                           # 20 features
            chroma_mean,                         # 12 features
            chroma_std,                          # 12 features
            [spectral_centroid_mean],            # 1 feature
            [spectral_centroid_std],             # 1 feature
            [spectral_rolloff_mean],             # 1 feature
            [spectral_rolloff_std],              # 1 feature
            [spectral_bandwidth_mean],           # 1 feature
            [spectral_bandwidth_std],            # 1 feature
            [zcr_mean],                          # 1 feature
            [zcr_std],                           # 1 feature
            [tempo],                             # 1 feature
            [rms_mean],                          # 1 feature
            [rms_std],                           # 1 feature
            mel_spec_mean,                       # 40 features
            mel_spec_std                         # 40 features
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def preprocess_dataset(data_path):
    """Preprocess GTZAN dataset"""
    features = []
    labels = []
    
    # GTZAN dataset structure: genres_original/genre/song.wav
    genres = os.listdir(data_path)
    
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path):
            continue
        
        print(f"Processing {genre}...")
        file_count = 0
        
        for filename in os.listdir(genre_path):
            if filename.endswith('.wav') or filename.endswith('.au'):
                file_path = os.path.join(genre_path, filename)
                feature = extract_features(file_path)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(genre)
                    file_count += 1
        
        print(f"  - Processed {file_count} files")
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for genre, count in zip(unique, counts):
        print(f"  {genre}: {count}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Save preprocessed data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nDataset preprocessed:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {label_encoder.classes_}")
    
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    # Change this path to your GTZAN dataset location
    DATA_PATH = "Data/genres_original"
    
    if os.path.exists(DATA_PATH):
        preprocess_dataset(DATA_PATH)
    else:
        print(f"Please download GTZAN dataset and extract to {DATA_PATH}")
        print("Dataset link: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")