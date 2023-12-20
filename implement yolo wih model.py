import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLO model
yolo = YOLO('yolov8n.pt')

# Load the classification model
classification_model = load_model('best_model(1).h5')

# Create an ImageDataGenerator object
datagen = ImageDataGenerator()

# Open the video file
cap = cv2.VideoCapture(r"C:\Users\omarm\Desktop\tpp\20231203_102328.mp4")

# Initialize frame counter
cnt = 0

# Loop until the end of the video
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was properly read
    if ret == True:
        # Save every 15th frame (for 0.5 second intervals at 30fps video)
        if cnt % 30 == 0:
            # Predict the objects in the frame using YOLO
            results = yolo.predict(frame)

            # Initialize an Annotator object for the frame
            annotator = Annotator(frame)

            # Loop over each detected object
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Check if the object is a person
                    if yolo.names[int(box.cls)] == 'person':
                        # Get the bounding box coordinates
                        b = box.xyxy[0]  # format: (left, top, right, bottom)

                        # Crop the image to get only the bounding box
                        cropped_img = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                        cropped_img_resized = cv2.resize(cropped_img, (640, 480))

                        # Save the cropped image
                        frame_path = f'frame{cnt}.jpg'
                        cv2.imwrite(frame_path, cropped_img_resized)

                        # Create a dataframe containing the image file name
                        df = pd.DataFrame({'filename': [frame_path]})

                        # Use flow_from_dataframe to load and preprocess the image
                        img_gen = datagen.flow_from_dataframe(df, 
                                                              directory='.', 
                                                              x_col='filename', 
                                                              class_mode=None, 
                                                              target_size=(640, 480), 
                                                              batch_size=1, 
                                                              shuffle=False)

                        # Make a prediction using the classification model
                        predictions = classification_model.predict(img_gen)

                        # Get the class with the highest probability
                        predicted_class = np.argmax(predictions[0])

                        # Annotate the frame with the bounding box and the predicted class
                        annotator.box_label(box.xyxy[0], label=str(predicted_class))

                        # Delete the cropped image file
                        os.remove(frame_path)

            # Display the annotated frame
            plt.imshow(cv2.cvtColor(annotator.im, cv2.COLOR_BGR2RGB))
            plt.show()

        # Increment the frame counter
        cnt += 1

    # Break the loop if the video has ended
    else:
        break

# Release the video capture object
cap.release()
