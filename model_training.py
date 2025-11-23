import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load preprocessed data"""
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, label_encoder

def train_model():
    """Train Random Forest model with improved parameters"""
    print("Loading data...")
    X_train, X_test, y_train, y_test, label_encoder = load_data()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Scale features - important for consistent predictions
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest model...")
    # Improved Random Forest parameters for better genre discrimination
    model = RandomForestClassifier(
        n_estimators=200,          # More trees for better accuracy
        max_depth=30,              # Deeper trees to capture complex patterns
        min_samples_split=2,       # Allow finer splits
        min_samples_leaf=1,        # More granular leaves
        max_features='sqrt',       # Use sqrt of features per split
        bootstrap=True,
        class_weight='balanced',   # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Training accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Display confusion matrix with genre names
    genres = label_encoder.classes_
    cm_display = pd.DataFrame(cm, index=genres, columns=genres)
    print(cm_display)
    
    # Identify problematic pairs
    print("\nMost Confused Genre Pairs:")
    confusion_pairs = []
    for i in range(len(genres)):
        for j in range(len(genres)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((genres[i], genres[j], cm[i, j]))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_genre, pred_genre, count in confusion_pairs[:10]:
        print(f"  {true_genre} → {pred_genre}: {count} misclassifications")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'genre_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved!")
    
    # Feature importance analysis
    print("\nTop 20 Most Important Features:")
    feature_names = (
        [f'mfcc_mean_{i}' for i in range(20)] +
        [f'mfcc_std_{i}' for i in range(20)] +
        [f'chroma_mean_{i}' for i in range(12)] +
        [f'chroma_std_{i}' for i in range(12)] +
        ['spectral_centroid_mean', 'spectral_centroid_std'] +
        ['spectral_rolloff_mean', 'spectral_rolloff_std'] +
        ['spectral_bandwidth_mean', 'spectral_bandwidth_std'] +
        ['zcr_mean', 'zcr_std'] +
        ['tempo'] +
        ['rms_mean', 'rms_std'] +
        [f'mel_spec_mean_{i}' for i in range(40)] +
        [f'mel_spec_std_{i}' for i in range(40)]
    )
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(20, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Try to plot confusion matrix
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=genres, yticklabels=genres)
        plt.title('Confusion Matrix')
        plt.ylabel('True Genre')
        plt.xlabel('Predicted Genre')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
    except:
        print("\nCouldn't create confusion matrix plot (matplotlib might not be available)")
    
    return model, scaler, label_encoder, test_accuracy

if __name__ == "__main__":
    try:
        import pandas as pd
        train_model()
    except FileNotFoundError:
        print("Please run data_preprocessing.py first to create the training data!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()