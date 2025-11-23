import streamlit as st
import tempfile
import os
from prediction import GenreClassifier
import pandas as pd

# Page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

@st.cache_resource
def load_classifier():
    """Load the classifier (cached for performance)"""
    return GenreClassifier()

def main():
    st.title("🎵 Music Genre Classifier")
    st.write("Upload an audio file and get AI-powered genre predictions!")
    
    # Initialize classifier
    classifier = load_classifier()
    
    if classifier.model is None:
        st.error("Model not found! Please run model_training.py first.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        
        # Play audio
        st.audio(uploaded_file, format='audio/wav')
        
        # Predict button
        if st.button("🎯 Predict Genre", type="primary"):
            with st.spinner("Analyzing audio... This may take a few seconds."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    # Make prediction
                    genre, confidence, top_genres = classifier.predict_genre(temp_path)
                    
                    if genre:
                        # Display results
                        st.success("Prediction Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Genre", genre)
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                        
                        # Top 3 predictions
                        st.subheader("Top 3 Predictions")
                        
                        df = pd.DataFrame([
                            {"Rank": i+1, "Genre": g, "Probability": f"{p:.1%}"}
                            for i, (g, p) in enumerate(top_genres)
                        ])
                        
                        st.dataframe(df, hide_index=True, use_container_width=True)
                        
                        # Visualization
                        chart_data = pd.DataFrame({
                            'Genre': [g for g, _ in top_genres],
                            'Probability': [p for _, p in top_genres]
                        })
                        
                        st.bar_chart(chart_data.set_index('Genre'))
                        
                    else:
                        st.error("Failed to analyze the audio file. Please try a different file.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.write("This AI model can classify music into the following genres:")
        
        if classifier.model is not None:
            genres = classifier.label_encoder.classes_
            for genre in sorted(genres):
                st.write(f"• {genre.title()}")
        
        st.header("Tips")
        st.write("• Use clear, good quality audio files")
        st.write("• 30-second clips work best")
        st.write("• Supported formats: WAV, MP3, FLAC, M4A")
        
        st.header("Model Info")
        st.write("• Algorithm: Random Forest")
        st.write("• Features: MFCC (13 coefficients)")
        st.write("• Dataset: GTZAN Music Genre Dataset")

if __name__ == "__main__":
    main()