import argparse
import wandb
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist
import numpy as np

def get_dataset(name):
    datasets = {
        "mnist": mnist,
        "fashion_mnist": fashion_mnist
    }
    return datasets[name].load_data()

def log_sample_images(images, labels, name):
    dataset_class_names = {
        "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "fashion_mnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    }
    class_names = dataset_class_names[name]
    selected_images = []
    selected_labels = []
    for i in range(10):  # 10 classes
        idx = np.where(labels == i)[0][0]  # Get the first occurrence of each class
        selected_images.append(images[idx])
        selected_labels.append(class_names[i])
    
    wandb.log({
        "examples": [wandb.Image(img, caption=label) for img, label in zip(selected_images, selected_labels)]
    })


class NeuralNetwork:
    def __init__(self, args):
        self.args = args
        weights = [np.random.rand(28*28, args.hidden_size)]
        biases = [np.random.rand(args.hidden_size)]   
        for i in range(args.num_layers-1):
            weights.append(np.random.rand(args.hidden_size,args.hidden_size))
            biases.append(np.random.rand(args.hidden_size))
        weights.append(np.random.rand(args.hidden_size,10))
        biases.append(np.random.rand(10))
        self.weights = weights
        self.biases = biases


    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward(self, x):
        for i in range(self.args.num_layers):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            if self.args.activation == "sigmoid":
                x = 1/(1+np.exp(-x))
            elif self.args.activation == "tanh":
                x = np.tanh(x)
            elif self.args.activation == "ReLU":
                x = np.maximum(0,x)
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        return self.softmax(x)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="Assignment1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="harshayelchuri-indian-institute-of-technology-madras")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(x_train.shape[0], 28*28), x_test.reshape(x_test.shape[0], 28*28)
    # log_sample_images(x_train, y_train, args.dataset)

    model = NeuralNetwork(args)
    print(x_train[0].shape, np.resize(x_train[0],(1,28*28)).shape)
    print(model.forward(x_train[0]))

    
    wandb.finish()




if __name__ == "__main__":
    main()