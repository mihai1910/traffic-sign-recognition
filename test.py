import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Setup the video capture and model
frameWidth = 640  # Camera resolution
frameHeight = 480
brightness = 180
threshold = 0.75  # Probability threshold
font = cv2.FONT_HERSHEY_SIMPLEX

# Set up the video camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Import the trained model
model = load_model("my_model.keras")  # Load the Keras model

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Get class name based on the index
def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
        'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
        'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
        'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
        'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
        'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo] if 0 <= classNo < len(class_names) else "Unknown Class"

# Main loop
while True:
    # Read image from the camera
    success, imgOriginal = cap.read()

    # Preprocess the image for prediction
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)  # Reshape for the model

    # Prediction process
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)  # Get the class index with the highest probability
    probabilityValue = np.max(predictions)  # Get the max probability

    # Display predictions on the original image
    cv2.putText(imgOriginal, "CLASS: " + getClassName(classIndex), (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the result image
    cv2.imshow("Result", imgOriginal)

    # Break the loop if the user presses 'q'
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the camera and close windows
cv2.destroyAllWindows()
cap.release()
