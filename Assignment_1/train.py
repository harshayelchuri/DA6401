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
        if self.args.weight_init == "random":
            self.weights, self.biases = self.random_init(self.args)
        elif self.args.weight_init == "Xavier":
            self.weights, self.biases = self.xavier_init()
        self.initialize_gradients()


    def random_init(self, args):
        scale = 0.1  
        weights = [np.random.uniform(-scale, scale, (28*28, args.hidden_size))]
        biases = [np.zeros(args.hidden_size)]  
        for _ in range(args.num_layers - 1):
            weights.append(np.random.uniform(-scale, scale, (args.hidden_size, args.hidden_size)))
            biases.append(np.zeros(args.hidden_size))  
        weights.append(np.random.uniform(-scale, scale, (args.hidden_size, 10)))
        biases.append(np.zeros(10)) 
        return weights, biases
     

    def xavier_init(self):
        lower_bound = -np.sqrt(6 / (28 * 28 + self.args.hidden_size))
        upper_bound = np.sqrt(6 / (28 * 28 + self.args.hidden_size))
        weights = [np.random.uniform(lower_bound, upper_bound, (28 * 28, self.args.hidden_size))]
        biases = [np.zeros(self.args.hidden_size)]
        for i in range(self.args.num_layers - 1):
            lower_bound = -np.sqrt(6 / (self.args.hidden_size + self.args.hidden_size))
            upper_bound = np.sqrt(6 / (self.args.hidden_size + self.args.hidden_size))
            weights.append(np.random.uniform(lower_bound, upper_bound, (self.args.hidden_size, self.args.hidden_size)))
            biases.append(np.zeros(self.args.hidden_size))
        lower_bound = -np.sqrt(6 / (self.args.hidden_size + 10))
        upper_bound = np.sqrt(6 / (self.args.hidden_size + 10))
        weights.append(np.random.uniform(lower_bound, upper_bound, (self.args.hidden_size, 10)))
        biases.append(np.zeros(10))
        return weights, biases

    def initialize_gradients(self):
        self.gradients_w = [np.zeros_like(w) for w in self.weights]
        self.gradients_b = [np.zeros_like(b) for b in self.biases]
        # self.gradients_w = []
        # self.gradients_b = []
        # self.gradients_w.append(np.zeros((28 * 28, self.args.hidden_size)))
        # self.gradients_b.append(np.zeros(self.args.hidden_size))
        # for _ in range(self.args.num_layers - 1):
        #     self.gradients_w.append(np.zeros((self.args.hidden_size, self.args.hidden_size)))
        #     self.gradients_b.append(np.zeros(self.args.hidden_size))
        # self.gradients_w.append(np.zeros((self.args.hidden_size, 10)))
        # self.gradients_b.append(np.zeros(10))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)
    
    def ReLU(self, x):
        return np.maximum(0, x)

    def identity(self, x):
        return x

    def apply_activation(self, x):
        if self.args.activation == "sigmoid":
            return self.sigmoid(x)
        elif self.args.activation == "tanh":
            return self.tanh(x)
        elif self.args.activation == "ReLU":
            return self.ReLU(x)
        elif self.args.activation == "identity":
            return self.identity(x)

    def gradient_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def gradient_tanh(self, x):
        return 1 - np.square(self.tanh(x))
    
    def gradient_ReLU(self, x):
        return np.where(x > 0, 1, 0)

    def gradient_identity(self, x):
        return np.ones_like(x)
    
    def loss(self, y_true, y_pred):
        if self.args.loss == "mean_squared_error":
            return np.mean(np.square(y_true - y_pred))
        elif self.args.loss == "cross_entropy":
            # epsilon = 1e-9  
            # return -np.sum(y_true * np.log(np.clip(y_pred, epsilon, 1)))
            return -np.sum(y_true * np.log(y_pred))
        
    def forward(self, z):
        self.preactivations = []
        self.activations = []
        for i in range(len(self.weights) - 1):
            z = np.dot(z, self.weights[i]) + self.biases[i]
            self.preactivations.append(z)
            z = self.apply_activation(z)
            self.activations.append(z) 
        z = np.dot(z, self.weights[-1]) + self.biases[-1]
        self.preactivations.append(z)
        if self.args.loss == "cross_entropy":
            z = self.softmax(z)  
        self.activations.append(z)
        return z


    def backward(self, y_true, y_pred):
        self.initialize_gradients()
        loss = self.loss(y_true, y_pred)
        if self.args.loss == "cross_entropy":
            self.gradient_preactivation = y_pred - y_true
        elif self.args.loss == "mean_squared_error":
            pass

        for i in range(len(self.weights) - 1, -1, -1):
            self.gradients_w[i] = np.dot(self.activations[i].T, self.gradient_preactivation)
            self.gradients_b[i] = self.gradient_preactivation
            if i>0:
                self.gradient_activation = np.dot(self.gradient_preactivation, self.weights[i].T)
                if self.args.activation == "sigmoid":
                    self.gradient_preactivation = self.gradient_activation * self.gradient_sigmoid(self.preactivations[i - 1])
                elif self.args.activation == "tanh":
                    self.gradient_preactivation = self.gradient_activation * self.gradient_tanh(self.preactivations[i - 1])
                elif self.args.activation == "ReLU":
                    self.gradient_preactivation = self.gradient_activation * self.gradient_ReLU(self.preactivations[i - 1])   
                elif self.args.activation == "identity":
                    self.gradient_preactivation = self.gradient_activation * self.gradient_identity(self.preactivations[i - 1]) 
        
        # return self.gradients_w, self.gradients_b



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="Assignment1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="harshayelchuri-indian-institute-of-technology-madras")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
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
    parser.add_argument("-sz", "--hidden_size", type=int, default=32)
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(x_train.shape[0], 28*28), x_test.reshape(x_test.shape[0], 28*28)
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]
    # log_sample_images(x_train, y_train, args.dataset)

    model = NeuralNetwork(args)
    for epoch in range(args.epochs):
        print(f'epoch: {epoch}')
        temp_gradients_w = [np.zeros_like(w) for w in model.weights]
        temp_gradients_b = [np.zeros_like(b) for b in model.biases]
        epoch_loss = 0  
        correct_predictions = 0  
        for i in range(0, len(x_train)):
            y_pred = model.forward(x_train[i])
            loss = model.loss(y_train[i], y_pred)
            epoch_loss += loss
            if np.argmax(y_pred) == np.argmax(y_train[i]):
                correct_predictions += 1
            model.backward(y_train[i], y_pred)
            for j in range(len(model.weights)): 
                temp_gradients_w[j] += model.gradients_w[j]
                temp_gradients_b[j] += model.gradients_b[j]
            if (i + 1) % args.batch_size == 0 or (i + 1) == len(x_train):
                for j in range(len(model.weights)): 
                    model.weights[j] -= args.learning_rate * temp_gradients_w[j] / args.batch_size
                    model.biases[j] -= args.learning_rate * temp_gradients_b[j] / args.batch_size
                temp_gradients_w = [np.zeros_like(w) for w in model.weights]
                temp_gradients_b = [np.zeros_like(b) for b in model.biases]
        avg_epoch_loss = epoch_loss / len(x_train)
        train_accuracy  = correct_predictions / len(x_train)

        val_loss = 0
        val_correct_predictions = 0
        for i in range(len(x_test)):
            y_pred = model.forward(x_test[i])
            val_loss += model.loss(y_test[i], y_pred)
            if np.argmax(y_pred) == np.argmax(y_test[i]):
                val_correct_predictions += 1

        avg_val_loss = val_loss / len(x_test)
        val_accuracy = val_correct_predictions / len(x_test)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
        })
    
    
    wandb.finish()




if __name__ == "__main__":
    main()