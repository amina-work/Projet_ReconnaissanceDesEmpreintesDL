from PIL import Image
#import numpy as np
import os

# Open a TIF file
image = Image.open('database\\101_1.tif')

#importing the folder with the images
direct = 'C:\\Users\\DrdrA\\OneDrive\\Desktop\\UNI\\DeepLearning\\TP1\\database'

for fingerprint in os.listdir(direct):
    if fingerprint.endswith('.tif'):
        file_path = os.path.join(direct, fingerprint)
        image = Image.open(file_path)
        print(image)
        #process l'image

        #process fini
        image.close()