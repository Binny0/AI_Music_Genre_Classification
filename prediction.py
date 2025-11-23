import librosa
import numpy as np
import pickle
import joblib
from data_preprocessing import extract_features

class GenreClassifier:
    def __init__(self):
        """Initialize the classifier with trained model"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load trained model, scaler, and label encoder"""
        try:
            self.model = joblib.load('genre_classifier.pkl')
            self.scaler = joblib.load('scaler.pkl')
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Model loaded successfully!")
            print(f"Available genres: {', '.join(self.label_encoder.classes_)}")
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Please run model_training.py first!")
    
    def predict_genre(self, audio_path, show_details=True):
        """Predict genre for a single audio file with detailed output"""
        if self.model is None:
            return None, None, None
        
        print(f"\nAnalyzing: {audio_path}")
        
        # Extract features
        features = extract_features(audio_path)
        
        if features is None:
            print("Failed to extract features from audio file.")
            return None, None, None
        
        # Reshape and scale features
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get genre name and confidence
        genre = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        # Get top 5 predictions for better insight
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_genres = []
        
        for idx in top_indices:
            genre_name = self.label_encoder.classes_[idx]
            prob = probabilities[idx]
            top_genres.append((genre_name, prob))
        
        if show_details:
            print(f"\n{'='*50}")
            print(f"🎵 Predicted Genre: {genre.upper()}")
            print(f"📊 Confidence: {confidence*100:.2f}%")
            print(f"{'='*50}")
            print("\n📈 All Genre Probabilities:")
            for i, (g, prob) in enumerate(top_genres, 1):
                bar_length = int(prob * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"{i}. {g:12s} [{bar}] {prob*100:5.2f}%")
            
            # Warning if confidence is low
            if confidence < 0.4:
                print("\n⚠️  Warning: Low confidence prediction.")
                print("   The model is uncertain. Consider these possibilities:")
                for g, prob in top_genres[:3]:
                    print(f"   - {g}: {prob*100:.1f}%")
        
        return genre, confidence, top_genres
    
    def batch_predict(self, audio_paths):
        """Predict genres for multiple audio files"""
        results = []
        
        print(f"\nProcessing {len(audio_paths)} files...\n")
        
        for i, path in enumerate(audio_paths, 1):
            print(f"[{i}/{len(audio_paths)}]", end=" ")
            genre, confidence, top_genres = self.predict_genre(path, show_details=False)
            
            results.append({
                'file': path,
                'predicted_genre': genre,
                'confidence': confidence,
                'top_3': top_genres[:3] if top_genres else []
            })
            
            if genre:
                print(f"✓ {path} → {genre} ({confidence*100:.1f}%)")
            else:
                print(f"✗ {path} → Failed")
        
        return results
    
    def predict_with_audio_analysis(self, audio_path):
        """Predict with additional audio analysis for debugging"""
        print(f"\n{'='*60}")
        print(f"Detailed Audio Analysis: {audio_path}")
        print(f"{'='*60}")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=30, sr=22050)
            
            # Basic audio properties
            duration = librosa.get_duration(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            print(f"\n📊 Audio Properties:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Tempo: {tempo:.1f} BPM")
            print(f"   Spectral Centroid: {spectral_centroid:.1f} Hz")
            print(f"   Zero Crossing Rate: {zcr:.4f}")
            
            # Genre-specific characteristics
            print(f"\n🎼 Genre Indicators:")
            if tempo < 100:
                print(f"   - Slow tempo ({tempo:.0f} BPM) → Blues, Jazz possible")
            elif tempo > 140:
                print(f"   - Fast tempo ({tempo:.0f} BPM) → Metal, Disco, Electronic possible")
            else:
                print(f"   - Medium tempo ({tempo:.0f} BPM) → Rock, Pop, Country possible")
            
            if spectral_centroid > 3000:
                print(f"   - Bright sound → Metal, Electronic possible")
            elif spectral_centroid < 1500:
                print(f"   - Warm sound → Blues, Jazz, Classical possible")
            
        except Exception as e:
            print(f"Error in audio analysis: {e}")
        
        # Make prediction
        return self.predict_genre(audio_path, show_details=True)

def main():
    """Test the classifier with enhanced interface"""
    print("="*60)
    print("🎵 Music Genre Classifier")
    print("="*60)
    
    classifier = GenreClassifier()
    
    if classifier.model is None:
        return
    
    print("\nOptions:")
    print("  1. Enter audio file path")
    print("  2. Detailed analysis mode (shows audio characteristics)")
    print("  Type 'quit' to exit")
    
    while True:
        print("\n" + "-"*60)
        choice = input("\nEnter your choice (1/2) or file path: ").strip()
        
        if choice.lower() == 'quit':
            print("Goodbye! 👋")
            break
        
        if choice == '1':
            audio_file = input("Enter path to audio file: ").strip()
            if audio_file:
                classifier.predict_genre(audio_file)
        
        elif choice == '2':
            audio_file = input("Enter path to audio file for detailed analysis: ").strip()
            if audio_file:
                classifier.predict_with_audio_analysis(audio_file)
        
        else:
            # Assume it's a file path
            if choice:
                classifier.predict_genre(choice)

if __name__ == "__main__":
    main()