import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image

# Ensure TensorFlow optimization options are disabled
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the machine learning model
try:
    model = load_model(r'my_model.keras')
except OSError:
    st.error("Model file not found. Please check the path and try again.")
    st.stop()

# Function to predict autism
def predict_autism(image):
    img = keras_image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        return 'Non-Autistic'
    else:
        return 'Autistic'

def main():
    st.title('Autism Detection')
    st.write("This application detects autism from uploaded images or captured live pictures.")

    # Retrieve the questionnaire result from session state
    questionnaire_result = st.session_state.get('prediction', None)

    if questionnaire_result:
        st.write(f"Prediction from questionnaire: **{questionnaire_result}**")
    else:
        st.warning("No prediction from the questionnaire available. Please complete the questionnaire first.")

    # Option to upload image or capture from webcam
    option = st.radio("Select Option:", ("Upload Image", "Capture Live Picture"))

    image_prediction = None  # Initialize image prediction

    if option == "Upload Image":
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            image = img.resize((128, 128))

            st.write("Classifying...")
            image_prediction = predict_autism(image)
            st.write(f'The person in the image is predicted to be: **{image_prediction}**')

    if option == "Capture Live Picture":
        st.subheader("Capture Live Picture")
        st.write("Please press the button below to capture the live picture.")

        cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

        if not cap.isOpened():
            st.error("Error: Unable to open webcam.")
            return

        captured_image = None
        frame_placeholder = st.empty()
        capture_button = st.button("Capture")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to read frame from webcam.")
                break

            frame_placeholder.image(frame, channels="BGR", caption="Live Webcam Feed")

            if capture_button:
                captured_image = cv2.resize(frame, dsize=(128, 128))
                st.image(captured_image, channels="BGR", caption="Captured Image")
                break

        if captured_image is not None:
            st.write("Classifying...")
            image_prediction = predict_autism(captured_image)
            st.write(f'The person in the image is predicted to be: **{image_prediction}**')

        cap.release()

    # If both predictions (questionnaire + image) are present, compare and display results
    if st.button('Finalize Prediction'):
        if questionnaire_result and image_prediction:
            if questionnaire_result == "Autistic" and image_prediction == "Autistic":
                st.write("The final result is: **Autistic**")
                st.image(uploaded_file if option == "Upload Image" else captured_image, caption="Person Predicted as Autistic")
            elif questionnaire_result == "Non-Autistic" and image_prediction == "Non-Autistic":
                st.write("The final result is: **Non-Autistic**")
            elif questionnaire_result == "Autistic" and image_prediction == "Non-Autistic":
                st.write("The person is **70% likely to be Autistic** based on the questionnaire result.")
            elif questionnaire_result == "Non-Autistic" and image_prediction == "Autistic":
                st.write("The person is **70% likely to be Non-Autistic** based on the questionnaire result.")
        else:
            st.warning("You need predictions from both the questionnaire and image to finalize.")

if __name__ == "__main__":
    main()

# Footer
link_text = "Made with ❤️ by Gehena, Hithanksha, Hrithika, and Lavya"
link = f'<div style="display: block; text-align: center; margin-top:30%; padding: 5px;">{link_text}</div>'
st.write(link, unsafe_allow_html=True)
