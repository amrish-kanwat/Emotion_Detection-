import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
import plotly.graph_objects as go
from collections import deque
import os

# Try importing tensorflow with error handling
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not found. Please install tensorflow: `pip install tensorflow`")

# ------------------------------
# Configuration
# ------------------------------
class Config:
    MODEL_PATH = "emotion_cnn_model.h5"
    CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    EMOTION_COLORS = {
        'Angry': '#FF4136',
        'Disgust': '#85144b', 
        'Fear': '#B10DC9',
        'Happy': '#2ECC40',
        'Neutral': '#AAAAAA',
        'Sad': '#0074D9',
        'Surprise': '#FF851B'
    }
    TARGET_SIZE = (48, 48)
    CONFIDENCE_THRESHOLD = 0.3
    HISTORY_LENGTH = 50

# ------------------------------
# Model Loading with Error Handling
# ------------------------------
@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model with proper error handling."""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    try:
        if not os.path.exists(Config.MODEL_PATH):
            st.error(f"Model file '{Config.MODEL_PATH}' not found!")
            st.info("Please ensure your trained model file is in the same directory as this script.")
            return None
        
        model = load_model(Config.MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ------------------------------
# Enhanced Prediction Function
# ------------------------------
def predict_emotion(img, model):
    """Predict emotion from image with enhanced preprocessing."""
    if model is None:
        return "Error", 0.0
    
    try:
        # Convert to grayscale
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        # Resize to target size
        resized_img = cv2.resize(gray_img, Config.TARGET_SIZE)
        
        # Normalize and reshape
        img_array = resized_img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return Config.CLASS_LABELS[predicted_class_idx], confidence
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

# ------------------------------
# Face Detection
# ------------------------------
@st.cache_resource
def load_face_cascade():
    """Load OpenCV face cascade classifier."""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return None

def detect_faces(frame, face_cascade):
    """Detect faces in the frame."""
    if face_cascade is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    return faces

# ------------------------------
# UI Components
# ------------------------------
def create_emotion_chart(emotion_history):
    """Create a real-time emotion chart."""
    if not emotion_history:
        return go.Figure()
    
    emotions = list(Config.EMOTION_COLORS.keys())
    emotion_counts = {emotion: 0 for emotion in emotions}
    
    for emotion, _ in emotion_history:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(emotion_counts.keys()),
            y=list(emotion_counts.values()),
            marker_color=[Config.EMOTION_COLORS[emotion] for emotion in emotion_counts.keys()],
            text=list(emotion_counts.values()),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Detection History",
        xaxis_title="Emotions",
        yaxis_title="Count",
        height=300,
        showlegend=False
    )
    
    return fig

def draw_face_box_and_emotion(frame, faces, model):
    """Draw bounding boxes around faces and predict emotions."""
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face region for emotion prediction
        face_roi = frame[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_roi, model)
        
        # Choose color based on confidence
        color = (0, 255, 0) if confidence > Config.CONFIDENCE_THRESHOLD else (0, 165, 255)
        
        # Draw emotion label
        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return frame

# ------------------------------
# Main Application
# ------------------------------
def main():
    st.set_page_config(
        page_title="Advanced Emotion Detector", 
        page_icon="😊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎭 Advanced Real-Time Emotion Detection")
    st.markdown("**Detect emotions in real-time using your webcam with face detection!**")
    
    # Load models
    model = load_emotion_model()
    face_cascade = load_face_cascade()
    
    if model is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Settings
    show_face_detection = st.sidebar.checkbox("Enable Face Detection", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
    flip_camera = st.sidebar.checkbox("Flip Camera (Mirror Mode)", value=True)
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=Config.CONFIDENCE_THRESHOLD, 
        step=0.1
    )
    Config.CONFIDENCE_THRESHOLD = confidence_threshold
    
    # Camera selection
    camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        run_camera = st.checkbox("🎥 Start Camera", key="camera_toggle")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()
        chart_placeholder = st.empty()
    
    # Initialize session state
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = deque(maxlen=Config.HISTORY_LENGTH)
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Camera processing
    if run_camera:
        try:
            camera = cv2.VideoCapture(camera_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not camera.isOpened():
                st.error("Could not access camera. Please check camera permissions.")
                st.stop()
            
            fps_counter = 0
            start_time = time.time()
            
            while run_camera:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Flip frame if requested
                if flip_camera:
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = frame.copy()
                current_emotions = []
                
                if show_face_detection and face_cascade is not None:
                    # Detect faces and predict emotions for each
                    faces = detect_faces(frame, face_cascade)
                    
                    if len(faces) > 0:
                        processed_frame = draw_face_box_and_emotion(processed_frame, faces, model)
                        
                        # Get emotions for statistics
                        for (x, y, w, h) in faces:
                            face_roi = frame[y:y+h, x:x+w]
                            emotion, confidence = predict_emotion(face_roi, model)
                            if confidence > Config.CONFIDENCE_THRESHOLD:
                                current_emotions.append((emotion, confidence))
                                st.session_state.emotion_history.append((emotion, confidence))
                    else:
                        cv2.putText(processed_frame, "No faces detected", (30, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Predict emotion for entire frame
                    emotion, confidence = predict_emotion(frame, model)
                    if confidence > Config.CONFIDENCE_THRESHOLD:
                        current_emotions.append((emotion, confidence))
                        st.session_state.emotion_history.append((emotion, confidence))
                    
                    # Display prediction on frame
                    color = (0, 255, 0) if confidence > Config.CONFIDENCE_THRESHOLD else (0, 165, 255)
                    label = f"{emotion}: {confidence:.2f}" if show_confidence else emotion
                    cv2.putText(processed_frame, label, (30, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Update frame
                frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Update statistics
                fps_counter += 1
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    fps = fps_counter / (current_time - start_time)
                    
                    with stats_placeholder.container():
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("FPS", f"{fps:.1f}")
                            st.metric("Frames Processed", st.session_state.frame_count)
                        with col_b:
                            if current_emotions:
                                dominant_emotion = max(current_emotions, key=lambda x: x[1])
                                st.metric("Current Emotion", dominant_emotion[0])
                                st.metric("Confidence", f"{dominant_emotion[1]:.2f}")
                            else:
                                st.metric("Current Emotion", "None detected")
                    
                    # Update chart
                    if st.session_state.emotion_history:
                        fig = create_emotion_chart(st.session_state.emotion_history)
                        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"emotion_chart_{int(current_time)}")
                    
                    fps_counter = 0
                    start_time = current_time
                
                st.session_state.frame_count += 1
                time.sleep(0.03)  # Limit to ~30 FPS
            
            camera.release()
            
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
    
    else:
        frame_placeholder.info("📷 Click 'Start Camera' to begin emotion detection")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tips:**\n"
        "- Ensure good lighting\n"
        "- Face the camera directly\n"
        "- Keep a stable position\n"
        "- Check camera permissions"
    )

if __name__ == "__main__":
    main()