from PIL import Image
import random
import numpy as np
import os

# Open a TIF file
image = Image.open('database\\101_1.tif')

#importing the folder with the images
direct = 'C:\\Users\\DrdrA\\OneDrive\\Desktop\\UNI\\DeepLearning\\TP1\\database'

# Create empty lists for training and testing file paths
training_files = []
testing_files = []

for fingerprint in os.listdir(direct):
    if fingerprint.endswith('.tif'):
        file_path = os.path.join(direct, fingerprint)
        #for checking if it's works:
        image = Image.open(file_path)
        print(image)
        #using random.choices to add to the testing or training files:
        if random.choices([True, False], [0.8, 0.2])[0]:
            training_files.append(file_path)
        else:
            testing_files.append(file_path)
        #process l'image

        #process fini
        #image.close()

# Print the number of images in each set
print(f"Number of images in training set: {len(training_files)}")
print(f"Number of images in testing set: {len(testing_files)}")