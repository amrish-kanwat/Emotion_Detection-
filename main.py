import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
import plotly.graph_objects as go
from collections import deque
import os

try:
    from tensorflow.keras.models import load_model
    tf_available = True
except ImportError:
    tf_available = False
    st.error("TensorFlow not found. Install with: pip install tensorflow")

# Basic config
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
COLORS = {
    'Angry': '#FF4136', 'Disgust': '#85144b', 'Fear': '#B10DC9',
    'Happy': '#2ECC40', 'Neutral': '#AAAAAA', 'Sad': '#0074D9', 'Surprise': '#FF851B'
}
MODEL_FILE = "emotion_cnn_model.h5"

@st.cache_resource
def load_emotion_model():
    if not tf_available:
        return None
    
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file '{MODEL_FILE}' not found!")
        return None
    
    try:
        model = load_model(MODEL_FILE)
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_emotion(img, model):
    if model is None:
        return "Error", 0.0
    
    # Convert and preprocess
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 48, 48, 1)
    
    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred[0])
    conf = np.max(pred[0])
    
    return EMOTIONS[idx], conf

@st.cache_resource
def load_face_detector():
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        return None

def detect_faces(frame, cascade):
    if cascade is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    return faces

def create_emotion_chart(emotion_history):
    if not emotion_history:
        return go.Figure()
    
    counts = {emotion: 0 for emotion in EMOTIONS}
    for emotion, _ in emotion_history:
        if emotion in counts:
            counts[emotion] += 1
    
    fig = go.Figure([
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=[COLORS[e] for e in counts.keys()],
            text=list(counts.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Emotion History",
        height=300,
        showlegend=False
    )
    
    return fig

def process_faces(frame, faces, model, show_conf, threshold):
    current_emotions = []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_img = frame[y:y+h, x:x+w]
        emotion, conf = predict_emotion(face_img, model)
        
        if conf > threshold:
            current_emotions.append((emotion, conf))
            
        color = (0, 255, 0) if conf > threshold else (0, 165, 255)
        label = f"{emotion}: {conf:.2f}" if show_conf else emotion
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame, current_emotions

def main():
    st.set_page_config(
        page_title="Emotion Detector",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("🎭 Emotion Detection App")
    
    model = load_emotion_model()
    if model is None:
        st.stop()
    
    face_detector = load_face_detector()
    
    # Sidebar settings
    st.sidebar.header("Settings")
    use_face_detection = st.sidebar.checkbox("Face Detection", True)
    show_confidence = st.sidebar.checkbox("Show Confidence", True)
    flip_camera = st.sidebar.checkbox("Mirror Mode", True)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.1)
    camera_idx = st.sidebar.selectbox("Camera", [0, 1, 2], index=0)
    
    # Mode selection
    mode = st.radio("Mode:", ["📷 Image Upload", "🎥 Live Camera"], horizontal=True)
    
    if mode == "📷 Image Upload":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png', 'bmp'])
            
        with col2:
            st.subheader("Results")
            
        if uploaded_file:
            try:
                pil_image = Image.open(uploaded_file)
                img_array = np.array(pil_image)
                
                # Convert to OpenCV format
                if len(img_array.shape) == 3:
                    if img_array.shape[2] == 4:  # RGBA
                        opencv_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                    else:  # RGB
                        opencv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    opencv_img = img_array
                
                processed_img = opencv_img.copy()
                detected_emotions = []
                
                if use_face_detection and face_detector:
                    faces = detect_faces(opencv_img, face_detector)
                    
                    if faces.any():
                        for i, (x, y, w, h) in enumerate(faces):
                            cv2.rectangle(processed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            
                            face_roi = opencv_img[y:y+h, x:x+w]
                            emotion, confidence = predict_emotion(face_roi, model)
                            
                            detected_emotions.append({
                                'id': i+1,
                                'emotion': emotion,
                                'confidence': confidence
                            })
                            
                            label = f"Face {i+1}: {emotion}"
                            if show_confidence:
                                label += f" ({confidence:.2f})"
                            
                            cv2.putText(processed_img, label, (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        st.success(f"Found {len(faces)} face(s)")
                    else:
                        # No faces, analyze whole image
                        emotion, confidence = predict_emotion(opencv_img, model)
                        detected_emotions.append({
                            'id': 'Full Image',
                            'emotion': emotion,
                            'confidence': confidence
                        })
                        
                        label = f"Overall: {emotion}"
                        if show_confidence:
                            label += f" ({confidence:.2f})"
                        cv2.putText(processed_img, label, (30, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        st.warning("No faces detected")
                else:
                    # Analyze whole image
                    emotion, confidence = predict_emotion(opencv_img, model)
                    detected_emotions.append({
                        'id': 'Full Image',
                        'emotion': emotion,
                        'confidence': confidence
                    })
                    
                    label = f"Emotion: {emotion}"
                    if show_confidence:
                        label += f" ({confidence:.2f})"
                    cv2.putText(processed_img, label, (30, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display results
                col1.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                          caption="Processed Image", use_column_width=True)
                
                with col2:
                    if detected_emotions:
                        st.write("**Detected Emotions:**")
                        for result in detected_emotions:
                            st.write(f"**{result['id']}:** {result['emotion']} ({result['confidence']:.2f})")
                        
                        # Create chart
                        if len(detected_emotions) > 0:
                            emotions = [r['emotion'] for r in detected_emotions]
                            confidences = [r['confidence'] for r in detected_emotions]
                            
                            fig = go.Figure([
                                go.Bar(
                                    x=[f"Result {i+1}" for i in range(len(emotions))],
                                    y=confidences,
                                    text=emotions,
                                    textposition='auto',
                                    marker_color=[COLORS[e] for e in emotions]
                                )
                            ])
                            
                            fig.update_layout(
                                title="Detection Results",
                                height=300,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="image_results")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    else:
        # Live camera mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Camera")
            start_camera = st.checkbox("Start Camera")
            frame_area = st.empty()
        
        with col2:
            st.subheader("Stats")
            stats_area = st.empty()
            chart_area = st.empty()
        
        # Initialize session state for emotion tracking
        if 'emotion_history' not in st.session_state:
            st.session_state.emotion_history = deque(maxlen=50)
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        
        if start_camera:
            try:
                cap = cv2.VideoCapture(camera_idx)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if not cap.isOpened():
                    st.error("Can't access camera")
                    st.stop()
                
                fps_count = 0
                start_time = time.time()
                
                while start_camera:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if flip_camera:
                        frame = cv2.flip(frame, 1)
                    
                    processed_frame = frame.copy()
                    current_emotions = []
                    
                    if use_face_detection and face_detector:
                        faces = detect_faces(frame, face_detector)
                        
                        if len(faces) > 0:
                            processed_frame, current_emotions = process_faces(
                                processed_frame, faces, model, show_confidence, conf_threshold
                            )
                            
                            for emotion, conf in current_emotions:
                                st.session_state.emotion_history.append((emotion, conf))
                        else:
                            cv2.putText(processed_frame, "No faces detected", (30, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        emotion, conf = predict_emotion(frame, model)
                        if conf > conf_threshold:
                            current_emotions.append((emotion, conf))
                            st.session_state.emotion_history.append((emotion, conf))
                        
                        label = f"{emotion}: {conf:.2f}" if show_confidence else emotion
                        color = (0, 255, 0) if conf > conf_threshold else (0, 165, 255)
                        cv2.putText(processed_frame, label, (30, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    frame_area.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    
                    # Update stats every second
                    fps_count += 1
                    current_time = time.time()
                    if current_time - start_time >= 1.0:
                        fps = fps_count / (current_time - start_time)
                        
                        with stats_area.container():
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("FPS", f"{fps:.1f}")
                                st.metric("Frames", st.session_state.frame_count)
                            with col_b:
                                if current_emotions:
                                    top_emotion = max(current_emotions, key=lambda x: x[1])
                                    st.metric("Current", top_emotion[0])
                                    st.metric("Confidence", f"{top_emotion[1]:.2f}")
                                else:
                                    st.metric("Current", "None")
                        
                        # Update chart
                        if st.session_state.emotion_history:
                            fig = create_emotion_chart(st.session_state.emotion_history)
                            chart_area.plotly_chart(fig, use_container_width=True, key=f"live_chart_{int(current_time)}")
                        
                        fps_count = 0
                        start_time = current_time
                    
                    st.session_state.frame_count += 1
                    time.sleep(0.03)
                
                cap.release()
                
            except Exception as e:
                st.error(f"Camera error: {e}")
        else:
            frame_area.info("Click 'Start Camera' to begin")
    
    # Tips in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Tips:**\n"
        "• Good lighting helps accuracy\n"
        "• Face camera directly for best results\n"
        "• Multiple faces supported\n"
        "• Adjust confidence threshold as needed"
    )

if __name__ == "__main__":
    main()