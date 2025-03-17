import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
wandb.init(project="Assignment1",entity="harshayelchuri-indian-institute-of-technology-madras")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

selected_images = []
selected_labels = []
for i in range(10):  # 10 classes
    idx = np.where(y_train == i)[0][0]  # Get the first occurrence of each class
    selected_images.append(x_train[idx])
    selected_labels.append(class_names[i])

# Log images to wandb
wandb.log({
    "display images": [wandb.Image(img, caption=label) for img, label in zip(selected_images, selected_labels)]
})
wandb.alert(title="Assignment1", text="Images logged to wandb")
wandb.finish()
