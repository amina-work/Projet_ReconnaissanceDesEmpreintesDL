
from PIL import Image, ImageOps, ImageFilter  # Import ImageFilter
import random
import numpy as np
import os
from dask import layers
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import models
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, MaxPooling2D, Dropout
from sklearn.preprocessing import LabelEncoder
# Open a TIF files
image = Image.open('database\\101_1.tif')

#importing the folder with the images
direct = 'C:\\Users\\DrdrA\\OneDrive\\Desktop\\UNI\\M2\\DeepLearning\\projet\\database'

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
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')

# Optionally, you can save the lists of file paths to text files for future reference
# For example, you can save the training and testing file paths to text files
with open('training_files.txt', 'w') as file:
    file.writelines('\n'.join(training_files))

with open('testing_files.txt', 'w') as file:
    file.writelines('\n'.join(testing_files))

print("File paths saved to 'training_files.txt' and 'testing_files.txt'.")
