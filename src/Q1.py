# Load all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb

# Setup Wandb
wandb.login(key='5da0c161a9c9720f15195bb6e9f05e44c45112d1')
wandb.init(project="CS6910_AS1", entity='ed23s037')

# Load images
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Plot images
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fig, ax = plt.subplots(2,5)
ax = ax.flatten()
for i in range(10):
    index = np.argwhere(train_labels == i)[0]
    sample = np.reshape(train_images[index], (28, 28))
    ax[i].imshow(sample)
    wandb.log({"Sample Images": [wandb.Image(sample, caption=CLASS_NAMES[i])]})
wandb.finish()