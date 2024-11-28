from util import*
# import os
# import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
# import tensorflow as tf

# Reload model 
model = tf.keras.models.load_model('siamesemodel.keras', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy''Precision', 'Recall'])

#result on image
# test_input = preprocess('/home/luffy/Desktop/siamese/anchor/image2.jpg')
# test_val = preprocess('/home/luffy/Desktop/siamese/anchor/image2.jpg')
# test_input = np.expand_dims(test_input, axis=0)
# test_val = np.expand_dims(test_val, axis=0)

# Make predictions with reloaded model
# result = model.predict([test_input, test_val])
# print(result)

#realtime result

def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_image')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_image', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_image'))) 
    verified = verification > verification_threshold
    
    return results, verified

cap = None  # Global variable to manage the webcam

while True:
    # Taking input from the user
    user_input = input("Press 1 to enroll face, 2 to verify, or 0 to exit: ")

    if user_input == "1":
                #-----------capture video-------------
        # access web cam to capture video
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("unable to access camera")

        else:
            print("camera accessed successfully")

        
        enroll_face(20, pos_path="application_data/input_image", anchor_path="application_data/verification_image", pos_req=False, cap=cap)

    elif user_input == "2":
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Open the webcam if not already opened
        
        print("Press 'v' to verify, 'q' to quit verification mode.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                break

            # Cropping the frame (customize based on your requirement)
            frame = frame[120:120+250, 200:200+250, :]
            
            # Display the frame
            cv2.imshow('Verification', frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('v'):
                # Save input image to input_image folder
                input_image_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
                os.makedirs(os.path.dirname(input_image_path), exist_ok=True)
                cv2.imwrite(input_image_path, frame)
                
                # Run verification
                results, verified = verify(model, 0.4, 0.4)  # Replace `None` with your model
                print("Verified:", verified)
            
            elif key == ord('q'):
                break

    elif user_input == "0":
        print("Exiting program...")
        break
    else:
        print("Invalid input. Please enter 1, 2, or 0.")
    
    # Release webcam and close windows if opened
    if cap is not None and cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()

















