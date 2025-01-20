import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg for better compatibility with PyCharm
import matplotlib.pyplot as plt

# Define the path to your dataset
closed_eyes_path = './kaggle_data/train/Closed_Eyes'
open_eyes_path = './kaggle_data/train/Open_Eyes'

# Get the list of images in each class
closed_eyes_images = os.listdir(closed_eyes_path)
open_eyes_images = os.listdir(open_eyes_path)

# Display class distribution
print(f"Closed Eyes images: {len(closed_eyes_images)}")
print(f"Open Eyes images: {len(open_eyes_images)}")

# EXTRA Visualize some sample images
fig, ax = plt.subplots(1, 4, figsize=(12, 4))

# Display sample closed eyes image
img1 = cv2.imread(os.path.join(closed_eyes_path, closed_eyes_images[0]))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
ax[0].imshow(img1)
ax[0].set_title('Closed Eyes')
ax[0].axis('off')

# Display sample open eyes image
img2 = cv2.imread(os.path.join(open_eyes_path, open_eyes_images[0]))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
ax[1].imshow(img2)
ax[1].set_title('Open Eyes')
ax[1].axis('off')

# Display another closed eyes image
img3 = cv2.imread(os.path.join(closed_eyes_path, closed_eyes_images[1]))
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
ax[2].imshow(img3)
ax[2].set_title('Closed Eyes')
ax[2].axis('off')

# Display another open eyes image
img4 = cv2.imread(os.path.join(open_eyes_path, open_eyes_images[1]))
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
ax[3].imshow(img4)
ax[3].set_title('Open Eyes')
ax[3].axis('off')

plt.show()  # This should now display the images properly
