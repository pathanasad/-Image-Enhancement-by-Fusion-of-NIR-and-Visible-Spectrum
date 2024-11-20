 import cv2
import numpy as np
from skimage import filters

# Function to perform two-scale decomposition (Low-pass and High-pass filtering)
def two_scale_decomposition(image, sigma=1.0):
    # Low-pass filter (Gaussian Blur)
    low_pass = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # High-pass filter (Details)
    high_pass = image - low_pass
    return low_pass, high_pass

# Read visible and NIR images
visible_img = cv2.imread("visible_image.jpg", cv2.IMREAD_GRAYSCALE)
nir_img = cv2.imread("nir_image.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Two-Scale Decomposition to both images
low_vis, high_vis = two_scale_decomposition(visible_img)
low_nir, high_nir = two_scale_decomposition(nir_img)

# Fusion Strategy: Combine low-frequency components of NIR with high-frequency components of Visible
fused_low = low_nir
fused_high = high_vis

# Reconstruct the fused image
fused_img = fused_low + fused_high

# Show the results
cv2.imshow("Fused Image", fused_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Load pre-trained VGG16 model
base_model = VGG16(weights="imagenet", include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to preprocess image for deep learning model
def preprocess_img(img):
    img = cv2.resize(img, (224, 224))  # Resize to fit model input
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocess for VGG16
    return img

# Preprocess the visible and NIR images
visible_img_processed = preprocess_img(visible_img)
nir_img_processed = preprocess_img(nir_img)

# Extract features using VGG16
features_visible = model.predict(visible_img_processed)
features_nir = model.predict(nir_img_processed)

# Fusion of features (Simple concatenation in this example)
fused_features = np.concatenate((features_visible, features_nir), axis=-1)

# Further processing of fused features, e.g., classification or image enhancement, can be done
# For simplicity, here we just combine the processed low-frequency and high-frequency results.
enhanced_img = fused_img  # Assuming simple fusion, can be more complex depending on model output

# Convert the result to a suitable range for display (0 to 255)
enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)

# Show the enhanced image
cv2.imshow("Enhanced Image", enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
