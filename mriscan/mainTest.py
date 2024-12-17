import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainMRI10EpochsCategorical.keras')

# Read the image
image = cv2.imread('C:\\Users\\mestha\\Downloads\\Brain Tumor Classification\\pred\\pred5.jpg')

# Convert the image to RGB (OpenCV loads images in BGR format)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to PIL format
img = Image.fromarray(image)

# Resize the image to the required size
img = img.resize((64, 64))

# Convert the image to a numpy array
img = np.array(img)

# Expand dimensions to match the input shape of the model
input_img = np.expand_dims(img, axis=0)

# Predict the class probabilities
result = model.predict(input_img)

# Get the class with the highest probability
predicted_class = np.argmax(result, axis=1)

print(predicted_class)
