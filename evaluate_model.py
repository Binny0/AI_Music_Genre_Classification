import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support

def load_test_data():
    """Load test data and models"""
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    model = joblib.load('genre_classifier.pkl')
    scaler = joblib.load('scaler.pkl')
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return X_test, y_test, model, scaler, label_encoder

def evaluate_model():
    """Comprehensive model evaluation"""
    print("Loading model and test data...")
    X_test, y_test, model, scaler, label_encoder = load_test_data()
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Weighted F1-Score: {f1:.3f}")
    print()
    
    # Detailed classification report
    print("Classification Report:")
    print("-"*30)
    report = classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_,
        digits=3
    )
    print(report)
    
    # Per-class metrics
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    
    print("\nPer-Class Detailed Metrics:")
    print("-"*50)
    for i, genre in enumerate(label_encoder.classes_):
        print(f"{genre:12} | P: {precision[i]:.3f} | R: {recall[i]:.3f} | F1: {f1_scores[i]:.3f} | Support: {support[i]}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Genre')
    plt.xlabel('Predicted Genre')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = [f'MFCC_{i+1}' for i in range(len(feature_importance))]
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importance)[::-1]
    plt.bar(range(len(feature_importance)), feature_importance[indices])
    plt.title('Feature Importance (MFCC Coefficients)')
    plt.xlabel('MFCC Coefficient')
    plt.ylabel('Importance')
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prediction confidence analysis
    max_probabilities = np.max(y_pred_proba, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(max_probabilities, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Max Probability (Confidence)')
    plt.ylabel('Number of Samples')
    plt.axvline(np.mean(max_probabilities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_probabilities):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Genre-wise accuracy
    genre_accuracy = []
    for i, genre in enumerate(label_encoder.classes_):
        genre_mask = y_test == i
        genre_acc = accuracy_score(y_test[genre_mask], y_pred[genre_mask])
        genre_accuracy.append(genre_acc)
    
    plt.figure(figsize=(12, 6))
    plt.bar(label_encoder.classes_, genre_accuracy)
    plt.title('Per-Genre Accuracy')
    plt.xlabel('Genre')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, acc in enumerate(genre_accuracy):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('genre_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved: confusion_matrix.png, feature_importance.png, confidence_distribution.png, genre_accuracy.png")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    try:
        results = evaluate_model()
    except FileNotFoundError:
        print("Model files not found. Please run model_training.py first!")