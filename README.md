#  Fashion Recommendation System

A web-based fashion image recommendation app built using **TensorFlow**, **ResNet50**, and **Streamlit**. Upload a fashion image and get visually similar fashion items recommended in real time.

---

## 🚀 Features

- Upload any fashion image (JPG, JPEG, PNG)  
- Extract deep features using pretrained ResNet50 + GlobalMaxPooling  
- Find similar images with K-Nearest Neighbors (Euclidean distance)  
- Interactive, user-friendly web interface with Streamlit  
- Custom CSS styling for polished UI/UX  

---

## 📸 How It Works

1. User uploads an image through the sidebar.  
2. The image is processed and features are extracted via ResNet50 (without top layers).  
3. Features are normalized and compared to a precomputed database of image features.  
4. The 5 most similar fashion items are displayed with thumbnail previews.  

---

hu
