import streamlit as st
import numpy as np
import os
from PIL import Image
import cv2
from keras.models import load_model
from keras.preprocessing import image as keras_image
import tensorflow as tf

# Ensure TensorFlow optimization options are disabled
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Enable eager execution in TensorFlow to avoid the name scope issue
tf.config.run_functions_eagerly(True)

# Load the machine learning model
model_path = r'my_model.keras'

if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except OSError as e:
        st.error(f"Error loading model: {e}")
        st.stop()
else:
    st.error("Model file not found. Please check the path and try again.")
    st.stop()

# Function to predict autism from an image
def predict_autism(image):
    img = keras_image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction[0][0] < 0.5:
        return 'Non-Autistic'
    else:
        return 'Autistic'

# Multi-Step Process on Same Page
def main():
    st.title("Autism Detection System")

    # Step 1: Questionnaire
    st.header("Step 1: Complete the Questionnaire")
    questions = [
        "Does your child look at you when you call their name?",
        "Does your child point to indicate something?",
        "Does your child follow where you're looking?",
        "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
        "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
        "Does your child follow where you're looking?",
        "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)?",
        "Does your child ever stare at nothing or wander with no purpose?",
        "Does your child use simple gestures? (e.g. wave goodbye)?",
        "Does your child stare at nothing with no apparent purpose?"
    ]

    user_answers = []
    for i, question in enumerate(questions):
        answer = st.radio(question, ("Yes", "No"), key=f"q{i}")
        user_answers.append(1 if answer == "Yes" else 0)

    # Simple rule-based prediction based on questionnaire
    if sum(user_answers) > 1:  # Example: If 2 or more "Yes" answers, predict "Autistic"
        questionnaire_result = "Autistic"
    else:
        questionnaire_result = "Non-Autistic"

    if st.button("Submit Questionnaire"):
        st.session_state['questionnaire_result'] = questionnaire_result
        st.success(f"Questionnaire Prediction: **{questionnaire_result}**")

    # Step 2: Upload or Capture Image
    if 'questionnaire_result' in st.session_state:
        st.header("Step 2: Upload or Capture Image for Autism Detection")

        option = st.radio("Select Image Option:", ("Upload Image", "Capture Live Picture"))

        if option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image.', use_column_width=True)
                image = img.resize((128, 128))

                st.write("Classifying image...")
                image_prediction = predict_autism(image)
                st.session_state['image_prediction'] = image_prediction
                st.session_state['captured_image'] = img  # Store uploaded image
                st.write(f"Image Prediction: **{image_prediction}**")

        elif option == "Capture Live Picture":
            st.subheader("Capture Live Picture")

            # Use OpenCV to access the webcam
            cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

            if cap.isOpened():
                # Capture a single frame when the button is pressed
                if st.button("Capture"):
                    ret, frame = cap.read()
                    if ret:
                        captured_image = cv2.resize(frame, (128, 128))
                        st.image(captured_image, channels="BGR", caption="Captured Image")
                        st.write("Classifying...")

                        # Convert OpenCV BGR image to RGB before prediction
                        captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(captured_image_rgb)

                        image_prediction = predict_autism(pil_image)
                        st.session_state['image_prediction'] = image_prediction
                        st.session_state['captured_image'] = pil_image  # Store captured image

                        # Display captured image prediction
                        st.write(f'The person in the image is predicted to be: {image_prediction}')
                    else:
                        st.error("Failed to capture image. Please try again.")
            else:
                st.error("Error: Unable to open webcam.")

            cap.release()

    # Step 3: Final Prediction
    if st.button("Finalize Prediction"):
        if 'questionnaire_result' in st.session_state and 'image_prediction' in st.session_state:
            final_result = None
            if st.session_state['questionnaire_result'] == "Autistic" and st.session_state['image_prediction'] == "Autistic":
                final_result = "Autistic"
            elif st.session_state['questionnaire_result'] == "Non-Autistic" and st.session_state['image_prediction'] == "Non-Autistic":
                final_result = "Non-Autistic"
            elif st.session_state['questionnaire_result'] == "Autistic" and st.session_state['image_prediction'] == "Non-Autistic":
                final_result = "70% likely Autistic"
            elif st.session_state['questionnaire_result'] == "Non-Autistic" and st.session_state['image_prediction'] == "Autistic":
                final_result = "70% likely Non-Autistic"

            
            if 'captured_image' in st.session_state:
                st.image(st.session_state['captured_image'], caption="Captured Image", use_column_width=True)
            st.markdown(f"<h2 style='font-size: 24px;'>The Person in the image is finally Predicted to be: {final_result}</h2>", unsafe_allow_html=True)

        else:
            st.warning("Please complete both steps to finalize the prediction.")

if __name__ == "__main__":
    main()
