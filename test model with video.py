import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the model
best_model = load_model('best_model.h5')

# Create an ImageDataGenerator object
datagen = ImageDataGenerator()

# Open the video file
cap = cv2.VideoCapture(r"C:\Users\omarm\Downloads\407536961_6277025192397339_9142313526632673030_n.mp4")

# Initialize frame counter
cnt = 0

# Loop until the end of the video
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was properly read
    if ret == True:
        # Save every 15th frame (for 0.5 second intervals at 30fps video)
        if cnt % 15 == 0:
            # Save the frame as an image
            frame_path = f'frame{cnt}.jpg'
            cv2.imwrite(frame_path, frame)

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

            # Make a prediction
            predictions = best_model.predict(img_gen)

            # Get the class with the highest probability
            predicted_class = np.argmax(predictions[0])

            # Plot the image with its prediction
            img = plt.imread(frame_path)
            plt.imshow(img)
            plt.title(f'Predicted class: {predicted_class}')
            plt.show()
            os.remove(frame_path)

        # Increment the frame counter
        cnt += 1

    # Break the loop if the video has ended
    else:
        break

# Release the video capture object
cap.release()
