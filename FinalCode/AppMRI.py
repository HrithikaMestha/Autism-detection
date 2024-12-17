import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import streamlit as st

# Load the model
model = load_model('BrainMRI10EpochsCategorical.keras')


# Function to get class name
def get_className(classNo):
    if classNo == 0:
        return "Non Autistic"
    elif classNo == 1:
        return "Autistic"

# Function to get result
def getResult(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    class_idx = np.argmax(result, axis=1)[0]
    return class_idx

# Streamlit UI
st.title("Autism Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the file to a temporary directory
    temp_dir = "./uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Predict and display result
    if st.button("Predict"):
        value = getResult(file_path)
        result = get_className(value)
        st.write(f"Prediction: {result}")
