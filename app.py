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
    
for fingerprint in os.listdir(direct):
    if fingerprint.endswith('.tif'):
        file_path = os.path.join(direct, fingerprint)
        #for checking if it's works:
        image = Image.open(file_path)
        print(image)

        
        # Preprocess the image
        processed_image = preprocess_image(image)
        # For checking if it's working, you can display the processed image
        processed_image.show()
        
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

#  lists of file paths to text files for future reference
with open('training_files.txt', 'w') as file:
    file.writelines('\n'.join(training_files))

with open('testing_files.txt', 'w') as file:
    file.writelines('\n'.join(testing_files))

print("File paths saved to 'training_files.txt' and 'testing_files.txt'.")
