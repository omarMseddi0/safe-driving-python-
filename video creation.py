
# yaml videooowat
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLO model
yolo = YOLO('yolov8n.pt')

# Load the classification model
classification_model = load_model('best_model.h5')

# Create an ImageDataGenerator object
datagen = ImageDataGenerator()
       #"C:\Users\omarm\Downloads\ayoub.mp4"
# Open the video file
cap = cv2.VideoCapture(r"C:\Users\omarm\Desktop\tpp\20231203_102328.mp4")

# Initialize frame counter
cnt = 0
person_detected = 0
# Loop until the end of the video
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was properly read
    if ret == True:
        # Increment the frame counter
        cnt += 1

        # Save every 15th frame
         # Save every 15th frame
        if cnt % 10 == 0:
            # Predict the objects in the frame using YOLO
            results = yolo.predict(frame)

            # Initialize an Annotator object for the frame
            annotator = Annotator(frame)

            # Initialize a variable to store the largest bounding box
            largest_box = None
            largest_area = 0

            # Loop over each detected object
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Check if the object is a person
                    if yolo.names[int(box.cls)] == 'person':
                        # Get the bounding box coordinates
                        b = box.xyxy[0]  # format: (left, top, right, bottom)
                        person_detected = 1
                        # Calculate the area of the bounding box
                        area = (b[2] - b[0]) * (b[3] - b[1])

                        # If this bounding box is larger than the current largest bounding box, update the largest bounding box
                        if area > largest_area:
                            largest_box = box
                            largest_area = area
                    else :
                        person_detected = 0
            # If a person was detected, annotate the frame with the largest bounding box and the predicted label
            if largest_box is not None:
                # Get the bounding box coordinates
                b = largest_box.xyxy[0]  # format: (left, top, right, bottom)

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

                # Map the predicted class to the corresponding label
                labels = {0: 'does not have phone', 1: 'tolk to a phone ', 2: 'hold a phone ', 3: 'tolk to a phone ', 4: 'hold a phone '}
                predicted_label = labels[predicted_class]
           
                # Annotate the frame with the bounding box and the predicted label
                annotator.box_label(largest_box.xyxy[0], label=predicted_label)
                # Delete the cropped image file
                os.remove(frame_path)
                        # Rest of your code...

            # If a person was detected, display the annotated frame
            if person_detected == 1 :
                 cv2.imshow('Video', cv2.resize(annotator.im, (1920, 1080)))
                
            # If no person was detected, display the original frame
            else:
                cv2.imshow('Video', cv2.resize(frame, (1920, 1080)))

            # Break the loop qqqif the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the video caqqqqqqqqqqqqqqqqqqpture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
cv2.destroyAllWindows()
