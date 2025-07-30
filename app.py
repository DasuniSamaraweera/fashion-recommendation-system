import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
import streamlit as st

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .header {
        font-family: 'Arial', sans-serif;
        color: #1a2a44;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subheader {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
        margin-top: 15px;
    }
    .image-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0px; /* Removed padding to eliminate extra space */
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .image-container:hover {
        transform: scale(1.05);
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 14px;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    </style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="Fashion Recommendation System", page_icon="👗", layout="wide")

# Header
st.markdown('<h1 class="header">Fashion Recommendation System</h1>', unsafe_allow_html=True)
st.write("Upload an image to discover similar fashion items!")

# Load precomputed features and model
@st.cache_resource
def load_model_and_data():
    Image_features = pkl.load(open('Image_Features.pkl', 'rb'))
    filenames = pkl.load(open('Filenames.pkl', 'rb'))
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(Image_features)
    return model, Image_features, filenames, neighbors

model, Image_features, filenames, neighbors = load_model_and_data()

# Feature extraction function
def extract_features_from_images(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_expand_dim = np.expand_dims(img_array, axis=0)
        img_preprocess = preprocess_input(img_expand_dim)
        result = model.predict(img_preprocess).flatten()
        norm_result = result / np.linalg.norm(result)
        return norm_result
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Sidebar for file upload
with st.sidebar:
    st.markdown('<h2 class="subheader">Upload Your Image</h2>', unsafe_allow_html=True)
    upload_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    st.info("Supported formats: JPG, JPEG, PNG")

# Main content
if upload_file is not None:
    # Save uploaded file
    os.makedirs('upload', exist_ok=True)
    file_path = os.path.join('upload', upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    # Display uploaded image
    st.markdown('<h2 class="subheader">Uploaded Image</h2>', unsafe_allow_html=True)
    st.image(upload_file, width=300)

    # Process image and show recommendations
    with st.spinner("Processing your image..."):
        input_img_features = extract_features_from_images(file_path, model)
        if input_img_features is not None:
            distance, indices = neighbors.kneighbors([input_img_features])
            
            # Display recommended images
            st.markdown('<h2 class="subheader">Recommended Fashion Items</h2>', unsafe_allow_html=True)
            cols = st.columns(5, gap="medium")
            for i, col in enumerate(cols, 1):
                with col:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(filenames[indices[0][i]], width=150)
                    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Please upload an image to get recommendations.")