import streamlit as st
import cv2
import numpy as np
import pickle
import time
from skimage.feature import hog, local_binary_pattern

st.set_page_config(
    page_title="Bisindo Sign Language Classifier",
    page_icon="🤟",
    layout="wide"
)

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
N_CLUSTERS = 100  
MAX_FEATURES = 150  
TARGET_SIZE = (256, 256)

color_palette = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
    '#e67e22', '#34495e', '#f1c40f', '#16a085', '#27ae60', '#2980b9',
    '#8e44ad', '#c0392b', '#d35400', '#7f8c8d'
]
CLASS_COLORS = {CLASSES[i]: color_palette[i % len(color_palette)] for i in range(len(CLASSES))}

class RootSIFT:
    def __init__(self, sift=None):
        self.sift = sift

    def detectAndCompute(self, image, param):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            return keypoints, None
        descriptors /= (np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)
        return keypoints, descriptors.astype(np.float32)


class HOGDescriptor:
    def __init__(self, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def compute(self, image):
        features = hog(image, orientations=self.orientations,
                      pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block,
                      block_norm='L2-Hys', feature_vector=True)
        return features.astype(np.float32)


class LBPDescriptor:
    def __init__(self, radius=3, n_points=24):
        self.radius = radius
        self.n_points = n_points
    
    def compute(self, image):
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')
        n_bins = self.n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype(np.float32)


@st.cache_resource
def load_models():
    models = {}
    descriptors = {}
    
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    descriptors['SIFT'] = sift
    descriptors['RootSIFT'] = RootSIFT(sift=sift)
    descriptors['AKAZE'] = cv2.AKAZE_create()
    descriptors['ORB'] = cv2.ORB_create(nfeatures=MAX_FEATURES)
    descriptors['HOG'] = HOGDescriptor()
    descriptors['LBP'] = LBPDescriptor()
    
    model_files = {
        'SIFT': 'models/rps_classifier_sift_best.pkl',
        'RootSIFT': 'models/rps_classifier_rootsift_best.pkl',
        'AKAZE': 'models/rps_classifier_akaze_best.pkl',
        'ORB': 'models/rps_classifier_orb_best.pkl',
        'HOG': 'models/rps_classifier_hog_best.pkl',
        'LBP': 'models/rps_classifier_lbp_best.pkl'
    }
    
    for name, filepath in model_files.items():
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            models[name] = {
                'svc': model_data['svc'],
                'scaler': model_data.get('scaler', None),
                'kmeans': model_data.get('kmeans', None),
                'results': model_data.get('results', {})
            }
        except Exception as e:
            st.warning(f"Could not load {name} model: {e}")
    
    return models, descriptors


def preprocess_image(img):
    img_resized = cv2.resize(img, TARGET_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return img_resized, gray_blur


def create_histogram(descriptors, kmeans, n_clusters):
    if descriptors is None:
        return np.zeros(n_clusters)
    words = kmeans.predict(descriptors)
    hist, _ = np.histogram(words, bins=range(n_clusters + 1))
    return hist / (np.linalg.norm(hist) + 1e-7)


def extract_keypoints_image(img_resized, gray_blur, descriptor, desc_name):
    if desc_name in ['HOG', 'LBP']:
        return None, None
    
    keypoints, desc = descriptor.detectAndCompute(gray_blur, None)
    if keypoints is None or len(keypoints) == 0:
        return None, None
    
    img_kp = cv2.drawKeypoints(
        img_resized, keypoints, None, 
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_kp_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)
    
    return img_kp_rgb, len(keypoints)


def classify_image(gray_blur, descriptor, model_data, desc_name):
    svc = model_data['svc']
    scaler = model_data.get('scaler', None)
    kmeans = model_data['kmeans']
    
    if desc_name in ['HOG', 'LBP']:
        features = descriptor.compute(gray_blur)
    else:
        _, desc = descriptor.detectAndCompute(gray_blur, None)
        features = create_histogram(desc, kmeans, N_CLUSTERS)
    
    if scaler is not None:
        features = scaler.transform([features])[0]
    
    probs = svc.predict_proba([features])[0]
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx]
    
    prob_dict = {CLASSES[i]: probs[i] for i in range(len(CLASSES))}
    
    return pred_class, confidence, prob_dict


def display_probability_bars(prob_dict):
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, prob in sorted_probs:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.write(f"{CLASS_EMOJIS[class_name]} **{class_name.capitalize()}**")
        with col2:
            st.progress(prob)
        with col3:
            st.write(f"**{prob:.1%}**")


def main():
    st.title("Bisindo Sign Language Classifier (A-Z)")
    st.markdown("Classify Indonesian Sign Language hand gestures")
    st.markdown("---")
    
    with st.spinner("Loading models..."):
        models, descriptors = load_models()
    
    if not models:
        st.error("No models found. Please ensure model files are in the 'models/' directory.")
        return
    
    st.sidebar.header("Settings")
    
    available_descriptors = list(models.keys())
    selected_descriptor = st.sidebar.selectbox(
        "Select Descriptor",
        available_descriptors,
        help="Choose the feature descriptor for classification"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Descriptor Info")
    
    descriptor_info = {
        'SIFT': 'Scale-Invariant Feature Transform - robust to scale and rotation',
        'RootSIFT': 'L1-normalized SIFT with square root - improved matching performance',
        'AKAZE': 'Accelerated-KAZE - fast and efficient keypoint detection',
        'ORB': 'Oriented FAST and Rotated BRIEF - very fast binary descriptor',
        'HOG': 'Histogram of Oriented Gradients - captures edge/gradient structure',
        'LBP': 'Local Binary Patterns - captures texture information'
    }
    
    st.sidebar.info(descriptor_info.get(selected_descriptor, ""))
    
    if selected_descriptor in models:
        results = models[selected_descriptor].get('results', {})
        if 'mean_accuracy' in results:
            st.sidebar.metric(
                "Model Accuracy", 
                f"{results['mean_accuracy']:.1%}",
            )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a hand gesture image (A-Z sign language)"
        )
        
        use_camera = st.checkbox("Use Camera Instead")
        if use_camera:
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                uploaded_file = camera_image
    
    with col2:
        st.subheader("Classification Result")
        result_placeholder = st.empty()
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Could not read the image. Please try another file.")
            return
        
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Classifying..."):
            start_time = time.time()
            
            img_resized, gray_blur = preprocess_image(img)
            
            model_data = models[selected_descriptor]
            descriptor = descriptors[selected_descriptor]
            
            pred_class, confidence, prob_dict = classify_image(
                gray_blur, descriptor, model_data, selected_descriptor
            )
            
            kp_image, num_keypoints = extract_keypoints_image(
                img_resized, gray_blur, descriptor, selected_descriptor
            )
            
            inference_time = time.time() - start_time
        
        with result_placeholder.container():
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {CLASS_COLORS[pred_class]}20; border-radius: 10px; border: 2px solid {CLASS_COLORS[pred_class]};">
                <h1 style="color: {CLASS_COLORS[pred_class]};">{CLASSES[pred_class]} {pred_class.upper()}</h1>
                <h2>{confidence:.1%} Confidence</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            time_color = "green" if inference_time < 0.1 else "orange" if inference_time < 0.5 else "red"
            st.markdown(f"Inference Time: **:{time_color}[{inference_time*1000:.1f} ms]**")
            
            st.markdown("### Class Probabilities")
            display_probability_bars(prob_dict)
        
        st.subheader("Feature Detection Visualization")
        
        if selected_descriptor in ['HOG', 'LBP']:
            st.info(f"{selected_descriptor} is a global descriptor and doesn't produce keypoints. "
                   "It computes features over the entire image.")
            
            if selected_descriptor == 'HOG':
                from skimage.feature import hog
                _, hog_image = hog(
                    gray_blur, orientations=9,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2),
                    visualize=True,
                    block_norm='L2-Hys'
                )

                hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min() + 1e-7)

                col_hog1, col_hog2 = st.columns(2)
                with col_hog1:
                    st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                with col_hog2:
                    st.image(hog_image, caption="HOG Features", use_container_width=True)
        else:
            if kp_image is not None:
                col_kp1, col_kp2 = st.columns(2)
                with col_kp1:
                    st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                with col_kp2:
                    st.image(kp_image, caption=f"{selected_descriptor} Keypoints ({num_keypoints} detected)", use_container_width=True)
            else:
                st.warning("No keypoints detected in the image.")
    
    else:
        with result_placeholder.container():
            st.info("Upload an image or use camera to classify a sign language gesture (A-Z)")

if __name__ == "__main__":
    main()
