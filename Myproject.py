
rom keras.utils import to_categorical
from PIL import Image, ImageOps, ImageFilter  # Import ImageFilter
import random
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from dask import layers
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import models
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, MaxPooling2D, Dropout,Activation
from sklearn.preprocessing import LabelEncoder
# Open a TIF files
image = Image.open('database2\\101_1.tif')
#importing the folder with the images
direct = 'C:\\Users\\ELITEBOOK\\PycharmProjects\\DL\\database2'

# Create empty lists for training and testing file paths
training_files = []
testing_files = []

X = []  # Store preprocessed images
y = []  # Store labels

# Define preprocessing functions
def preprocess_image(image):
    # Resize the image to a specific width and height
    desired_width = 200
    desired_height = 200
    image = image.resize((desired_width, desired_height))
    # Apply Gaussian filtering for noise reduction
    image = image.filter(ImageFilter.GaussianBlur(radius=5))
    # Enhance fingerprint ridges and valleys using histogram equalization
    image = ImageOps.equalize(image)
    # Binarize the image to create a black and white binary image
    image = image.convert('1')  # '1' mode is binary (black and white)
    return image

label_encoder = LabelEncoder()
for fingerprint in os.listdir(direct):
    if fingerprint.endswith('.tif'):
        file_path = os.path.join(direct, fingerprint)
        # Preprocess the image
        image = Image.open(file_path)
        #print(image)
        processed_image = preprocess_image(image)
        # For checking if it's working, you can display the processed image
        #processed_image.show()
         # Assign labels and perform label encoding
        label = 'in' if 'in' in fingerprint else 'out'
        label_encoded = label_encoder.fit_transform([label])[0]
        # Append preprocessed image and label to lists
        X.append(np.array(processed_image).reshape((200, 200, 1)))
        y.append(label_encoded)
        #using random.choices to add to the testing or training files:
        if random.choices([True, False], [0.8, 0.2])[0]:
            training_files.append(file_path)
        else:
            testing_files.append(file_path)
        #image.close()
# Print the number of images in each set
print(f"Number of images in training set: {len(training_files)}")
print(f"Number of images in testing set: {len(testing_files)}")

# Convert the data and labels to NumPy arrays
X = np.array(X)
y = np.array(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flatten the output before the Dense layer
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=2)
y_test_one_hot = to_categorical(y_test, num_classes=2)
# Create an ImageDataGenerator with augmentation options
#un outil puissant pour l'augmentation de données
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
#Train the model
#model.fit(X_train, y_train_one_hot, epochs=5, batch_size=32)
# Fit the generator on your training data
#instructing the data generator to analyze the training data and
# adapt its augmentation parameters accordingly.
datagen.fit(X_train)
# Train the model using augmented data
model.fit(datagen.flow(X_train, y_train_one_hot, batch_size=32), epochs=2)
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
print(f'Test accuracy: {test_acc}')

# Optionally, you can save the lists of file paths to text files for future reference
# For example, you can save the training and testing file paths to text files
with open('training_files.txt', 'w') as file:
    file.writelines('\n'.join(training_files))

with open('testing_files.txt', 'w') as file:
    file.writelines('\n'.join(testing_files))

print("File paths saved to 'training_files.txt' and 'testing_files.txt'.")

# Optionally, you can save the lists of file paths to text files for future reference
# For example, you can save the training and testing file paths to text files
with open('training_files.txt', 'w') as file:
    file.writelines('\n'.join(training_files))

with open('testing_files.txt', 'w') as file:
    file.writelines('\n'.join(testing_files))

print("File paths saved to 'training_files.txt' and 'testing_files.txt'.")
