# DA6401 Assignment 1
This repository contains the code for Assignment 1 of the DA6401 course. The project includes four Python files, each with a specific purpose. Below are the descriptions and instructions on how to run each file.



**The wandb report is available here: https://api.wandb.ai/links/harshayelchuri-indian-institute-of-technology-madras/7vmjba26**

**The github repository is found here: https://github.com/harshayelchuri/DA6401/tree/main**

## **display_images_Question_1.py**
**Description:** This file contains the code for displaying and logging sample images from the dataset to wandb.

**How to run:**
```bash
python display_images_Question_1.py
```

## **train.py**
**Description:** This file contains the code for training a neural network on the MNIST or Fashion MNIST dataset. It's loaded with the hyperparameters that give the highest accuracy.

**How to run:**
```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```

## **confusion_matrix_Question_7.py**
**Description:** This file contains the code for training a neural network and generating a confusion matrix for the predictions on the test set.

**How to run:**
```bash
python confusion_matrix_Question_7.py --dataset fashion_mnist --epochs 10 --batch_size 32 --learning_rate 0.001 --optimizer adam
```

## **wandb_sweep.py**
**Description:** This file contains the code for performing hyperparameter sweeps using Weights and Biases (wandb).

**How to run:**
```bash
python wandb_sweep.py --dataset mnist --epochs 10 --batch_size 32 --learning_rate 0.001 --optimizer adam
```
 
