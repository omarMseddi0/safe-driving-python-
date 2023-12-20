import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Read the Excel file
df = pd.read_csv(r'C:\Users\omarm\Downloads\train\driver_imgs_list.csv')

df = df[df['classname'].isin(['c0', 'c2', 'c4', 'c1', 'c3','c5'])]

# Map the classnames to your labels and convert to string
label_map = {'c0': '0', 'c2': '2', 'c4': '4', 'c1': '1', 'c3': '3', 'c5': '0' }
df['label'] = df['classname'].map(label_map).astype('str')

# Split the data into a training set and a test set
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
def build_model():
    model = Sequential()
    
    model.add(Conv2D(8, (3, 3),1, activation='relu',input_shape=(640, 480, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    
    model.add(Conv2D(16, (3, 3), 1,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), 1,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), 1,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(5, activation='softmax')) 
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Create a data generator
datagen = ImageDataGenerator(
    validation_split=0.2222  # set validation split to get 70% training, 20% validation
)

# Load the training data
train_data = datagen.flow_from_dataframe(train_df, directory=r'C:\Users\omarm\Downloads\train\allphotos', x_col='img', y_col='label', target_size=(640, 480), batch_size=50, subset='training', class_mode='categorical')

# Load the validation data
val_data = datagen.flow_from_dataframe(train_df, directory=r'C:\Users\omarm\Downloads\train\allphotos', x_col='img', y_col='label', target_size=(640, 480), batch_size=50, subset='validation', class_mode='categorical')

checkpoint = ModelCheckpoint('best_model(1).h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model = build_model()

history = model.fit(train_data,
             epochs=16,
             validation_data=val_data,
             callbacks=[checkpoint]  # Pass the checkpoint to the fit function
             )

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the model


# For the test set, you can create a new ImageDataGenerator without the validation_split
test_datagen = ImageDataGenerator()
test_data = test_datagen.flow_from_dataframe(test_df, directory=r'C:\Users\omarm\Downloads\train\allphotos', x_col='img', y_col='label', target_size=(640, 480), batch_size=60, class_mode='categorical')

# Load the saved model
from tensorflow.keras.models import load_model
# Load the best model
best_model = load_model('best_model(1).h5')

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(test_data)

print(f'Test accuracy: {test_accuracy}')