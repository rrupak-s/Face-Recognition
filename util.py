import cv2
import os
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

import tensorflow as tf
import pandas as pd
import os

# Define a custom callback to log metrics
class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file='training_metrics.xlsx'):
        super(MetricsLoggerCallback, self).__init__()
        self.log_file = log_file
        self.metrics_history = []

    def on_epoch_end(self, epoch, logs=None):
        # Append metrics at the end of every epoch
        logs = logs or {}
        self.metrics_history.append({'epoch': epoch + 1, **logs})
        print(f"Epoch {epoch + 1} metrics: {logs}")
        
        # Save metrics to an Excel file
        df = pd.DataFrame(self.metrics_history)
        df.to_excel(self.log_file, index=False)

#-----------capture video-------------
# access web cam to capture video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("unable to access camera")

else:
    print("camera accessed successfully")

# capture image for anchor and positive
def enroll_face(img_no: int, pos_path:str, anchor_path:str, pos_req: bool):

    framecount=0
    os.makedirs(pos_path, exist_ok=True)
    os.makedirs(anchor_path, exist_ok=True)

    if pos_req:
        img_no = img_no*2
    while True:

        ret,frame=cap.read()
        if not ret:
            print("error: unable to capture frame")
            break

        cv2.imshow("video feed",frame[0:250,150:400]) # to visualize video 

        key = cv2.waitKey(1) & 0xFF 

        if key == ord('q'):
            break

        if key == ord('c'):
            while True:
                ret, frame = cap.read()  # Capture frame
                if not ret:
                    break
                # Save the frame with sequential names
                framecount +=1
                if framecount == img_no:
                    print('capture complete')
                    break

                if pos_req:
                    

                    if framecount % 2 == 0 :
                        image_name = os.path.join(anchor_path, f"image{framecount}.jpg")
                        cv2.imwrite(image_name, frame[0:250,150:400])
                        print(f"Saved {image_name}")
                        

                    elif framecount % 2 != 0:
                        image_name = os.path.join(pos_path, f"image{framecount}.jpg")
                        cv2.imwrite(image_name, frame[0:250,150:400])
                        print(f"Saved {image_name}")
                
                else:
                    
                        image_name = os.path.join(anchor_path, f"image{framecount}.jpg")
                        cv2.imwrite(image_name, frame[0:250,150:400])
                        print(f"Saved {image_name}")                    
                        

                # Display the frame while saving
                cv2.imshow("Video Feed", frame)

                # Break saving loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
    cap.release()
    cap.destroyALLWindows()


def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0
    
    # Return image
    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

#Siamese network architecture

def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        # print(f"Input embedding type: {type(input_embedding)}, shape: {input_embedding.shape}")
        # print(f"Validation embedding type: {type(validation_embedding)}, shape: {validation_embedding.shape}")
        if isinstance(input_embedding, list):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, list):
            validation_embedding = validation_embedding[0]
        return tf.math.abs(input_embedding - validation_embedding)



def make_siamese_model(): 
    embedding = make_embedding()
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    print(f"----this-----{embedding(input_image)}")
    print(f"----this-----{embedding(validation_image)}")
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

@tf.function
def train_step(batch,siamese_model,binary_cross_loss,opt):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # Return loss
    return loss

def train(data, EPOCHS,siamese_model,binary_cross_loss,opt):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch,siamese_model,binary_cross_loss,opt)
            progbar.update(idx+1)
        
        # # Save checkpoints
        # if epoch % 10 == 0: 
        #     checkpoint.save(file_prefix=checkpoint_prefix)
        #     epoch_var.assign(epoch)
