from util import*

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

pos_path = 'positive'
neg_path = 'negative'
anchor_path = 'anchor'

img_number = 300

EPOCHS = 50

#---------------------use this to collect anchor and positive dataset-----------------
# enroll_face(img_number,pos_path,anchor_path,pos_req=True)

# load data into TF dataset
anchor = tf.data.Dataset.list_files(anchor_path + '/*jpg')
anchor = anchor.take(img_number - 5)  # Limit it manually

# anchor=tf.data.Dataset.list_files(anchor_path+'/*jpg').take(img_number-5)
postive=tf.data.Dataset.list_files(pos_path+'/*jpg').take(img_number-5)
negative=tf.data.Dataset.list_files(neg_path+'/*jpg').take(img_number-5)

positives =tf.data.Dataset.zip((anchor,postive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives =tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data=positives.concatenate(negatives)

# # anchor_length = tf.data.experimental.cardinality(data).numpy() # 

# data preprocessing 
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training data (70%)
train_data = data.take(round(len(data)*0.7))
train_data = train_data.batch(16).prefetch(8)

# Validation data (20%)
val_data = data.skip(round(len(data)*0.7)).take(round(len(data)*0.2))
val_data = val_data.batch(16).prefetch(8)

# Test data (10%)
test_data = data.skip(round(len(data)*0.9))
test_data = test_data.batch(16).prefetch(8)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


siamese_model = make_siamese_model()

train(train_data, EPOCHS,siamese_model,binary_cross_loss,opt)

siamese_model.save('siamesemodel.keras')