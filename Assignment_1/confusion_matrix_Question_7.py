import argparse
import wandb
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist
import numpy as np
import json

# from sklearn.metrics import confusion_matrix


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

def preprocess_data(args):
    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(x_train.shape[0], 28*28), x_test.reshape(x_test.shape[0], 28*28)
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    validation_split = 0.1
    val_size = int(validation_split * len(x_train))
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:] 

    return x_train, y_train, x_val, y_val, x_test, y_test  


def train(args, x_train, y_train, x_val, y_val, x_test, y_test):
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    run_name = f"confusion_matrix"

    wandb.run.name = run_name

    model = NeuralNetwork(args)
    history_gradients_1_w = [np.zeros_like(w) for w in model.weights]
    history_gradients_1_b = [np.zeros_like(b) for b in model.biases]
    history_gradients_2_w = [np.zeros_like(w) for w in model.weights]
    history_gradients_2_b = [np.zeros_like(b) for b in model.biases]
    for epoch in range(args.epochs):
        print(f'epoch: {epoch}')
        temp_gradients_w = [np.zeros_like(w) for w in model.weights]
        temp_gradients_b = [np.zeros_like(b) for b in model.biases]
        t=0
        epoch_loss = 0  
        correct_predictions = 0  
        for i in range(0, len(x_train)):
            if args.optimizer == "sgd" or args.optimizer == "momentum" or args.optimizer == "rmsprop" or args.optimizer == "adam" or args.optimizer == "nadam": 
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
                    t+=1
                    for j in range(len(model.weights)): 
                        if args.optimizer == "sgd":
                            model.weights[j] -= args.learning_rate * temp_gradients_w[j] / args.batch_size + args.learning_rate * args.weight_decay * model.weights[j]
                            model.biases[j] -= args.learning_rate * temp_gradients_b[j] / args.batch_size + args.learning_rate * args.weight_decay * model.biases[j]
                        if args.optimizer == "momentum":
                            history_gradients_1_w[j] = args.momentum * history_gradients_1_w[j] + (temp_gradients_w[j] / args.batch_size)
                            history_gradients_1_b[j] = args.momentum * history_gradients_1_b[j] + (temp_gradients_b[j] / args.batch_size)   
                            model.weights[j] -= args.learning_rate * history_gradients_1_w[j]  + args.learning_rate * args.weight_decay * model.weights[j]
                            model.biases[j] -= args.learning_rate * history_gradients_1_b[j] + args.learning_rate * args.weight_decay * model.biases[j]
                        if args.optimizer == "rmsprop":
                            history_gradients_1_w[j] = args.beta * history_gradients_1_w[j] + (1 - args.beta) * np.square(temp_gradients_w[j] / args.batch_size)
                            history_gradients_1_b[j] = args.beta * history_gradients_1_b[j] + (1 - args.beta) * np.square(temp_gradients_b[j] / args.batch_size)
                            model.weights[j] -= args.learning_rate * (temp_gradients_w[j] / args.batch_size) / (np.sqrt(history_gradients_1_w[j]) + args.epsilon) + args.learning_rate * args.weight_decay * model.weights[j]
                            model.biases[j] -= args.learning_rate * (temp_gradients_b[j] / args.batch_size) / (np.sqrt(history_gradients_1_b[j]) + args.epsilon) + args.learning_rate * args.weight_decay * model.biases[j]
                        if args.optimizer == "adam":
                            history_gradients_1_w[j] = args.beta1 * history_gradients_1_w[j] + (1 - args.beta1) * (temp_gradients_w[j] / args.batch_size)
                            history_gradients_2_w[j] = args.beta2 * history_gradients_2_w[j] + (1 - args.beta2) * np.square(temp_gradients_w[j] / args.batch_size)
                            history_gradients_1_b[j] = args.beta1 * history_gradients_1_b[j] + (1 - args.beta1) * (temp_gradients_b[j] / args.batch_size)
                            history_gradients_2_b[j] = args.beta2 * history_gradients_2_b[j] + (1 - args.beta2) * np.square(temp_gradients_b[j] / args.batch_size)
                            m_hat_w = history_gradients_1_w[j] / (1 - np.power(args.beta1, t))
                            v_hat_w = history_gradients_2_w[j] / (1 - np.power(args.beta2, t))
                            m_hat_b = history_gradients_1_b[j] / (1 - np.power(args.beta1, t))
                            v_hat_b = history_gradients_2_b[j] / (1 - np.power(args.beta2, t))
                            model.weights[j] -= args.learning_rate * (m_hat_w / (np.sqrt(v_hat_w) + args.epsilon)) + args.learning_rate * args.weight_decay * model.weights[j]
                            model.biases[j] -= args.learning_rate * (m_hat_b / (np.sqrt(v_hat_b) + args.epsilon)) + args.learning_rate * args.weight_decay * model.biases[j]
                        if args.optimizer == "nadam":
                            history_gradients_1_w[j] = args.beta1 * history_gradients_1_w[j] + (1 - args.beta1) * (temp_gradients_w[j] / args.batch_size)
                            history_gradients_2_w[j] = args.beta2 * history_gradients_2_w[j] + (1 - args.beta2) * np.square(temp_gradients_w[j] / args.batch_size)
                            history_gradients_1_b[j] = args.beta1 * history_gradients_1_b[j] + (1 - args.beta1) * (temp_gradients_b[j] / args.batch_size)
                            history_gradients_2_b[j] = args.beta2 * history_gradients_2_b[j] + (1 - args.beta2) * np.square(temp_gradients_b[j] / args.batch_size)
                            m_hat_w = history_gradients_1_w[j] / (1 - np.power(args.beta1, t))
                            v_hat_w = history_gradients_2_w[j] / (1 - np.power(args.beta2, t))
                            m_hat_b = history_gradients_1_b[j] / (1 - np.power(args.beta1, t))
                            v_hat_b = history_gradients_2_b[j] / (1 - np.power(args.beta2, t))
                            model.weights[j] -= args.learning_rate / (np.sqrt(v_hat_w) + args.epsilon) * (args.beta1 * m_hat_w + ((1 - args.beta1) / (1 - np.power(args.beta1, t))) * (temp_gradients_w[j] / args.batch_size)) + args.learning_rate * args.weight_decay * model.weights[j]
                            model.biases[j] -= args.learning_rate / (np.sqrt(v_hat_b) + args.epsilon) * (args.beta1 * m_hat_b + ((1 - args.beta1) / (1 - np.power(args.beta1, t))) * (temp_gradients_b[j] / args.batch_size)) + args.learning_rate * args.weight_decay * model.biases[j]
                    temp_gradients_w = [np.zeros_like(w) for w in model.weights]
                    temp_gradients_b = [np.zeros_like(b) for b in model.biases]
            if args.optimizer == "nag":
                old_weights = [w.copy() for w in model.weights]
                old_biases = [b.copy() for b in model.biases]
                for j in range(len(model.weights)):
                    model.weights[j] -= args.momentum * history_gradients_1_w[j]
                    model.biases[j] -= args.momentum * history_gradients_1_b[j]
                y_pred = model.forward(x_train[i])
                loss = model.loss(y_train[i], y_pred)
                epoch_loss += loss
                if np.argmax(y_pred) == np.argmax(y_train[i]):
                    correct_predictions += 1
                model.backward(y_train[i], y_pred)
                model.weights = old_weights
                model.biases = old_biases
                for j in range(len(model.weights)):
                    temp_gradients_w[j] += model.gradients_w[j]
                    temp_gradients_b[j] += model.gradients_b[j]
                if (i + 1) % args.batch_size == 0 or (i + 1) == len(x_train):
                    for j in range(len(model.weights)):
                        history_gradients_1_w[j] = args.momentum * history_gradients_1_w[j] + (temp_gradients_w[j] / args.batch_size)
                        history_gradients_1_b[j] = args.momentum * history_gradients_1_b[j] + (temp_gradients_b[j] / args.batch_size)
                        model.weights[j] -= args.learning_rate * history_gradients_1_w[j] + args.learning_rate * args.weight_decay * model.weights[j]
                        model.biases[j] -= args.learning_rate * history_gradients_1_b[j] + args.learning_rate * args.weight_decay * model.biases[j]
                    temp_gradients_w = [np.zeros_like(w) for w in model.weights]
                    temp_gradients_b = [np.zeros_like(b) for b in model.biases]

        avg_epoch_loss = epoch_loss / len(x_train)
        train_accuracy  = correct_predictions / len(x_train)

        val_loss = 0
        val_correct_predictions = 0
        for i in range(len(x_val)):
            y_pred = model.forward(x_val[i])
            val_loss += model.loss(y_val[i], y_pred)
            if np.argmax(y_pred) == np.argmax(y_val[i]):
                val_correct_predictions += 1

        avg_val_loss = val_loss / len(x_val)
        val_accuracy = val_correct_predictions / len(x_val)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
        })
    

    y_preditction_list= []
    y_true_list = []
    y_prob_list = []
    dataset_class_names = {
        "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "fashion_mnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    }
    
    #### plotting confusion matrix ###
    test_correct_predictions = 0
    for i in range(len(x_test)):
        y_pred = model.forward(x_test[i])
        if np.argmax(y_pred) == np.argmax(y_test[i]):
                test_correct_predictions += 1
        y_prob_list.append(y_pred)
        y_preditction_list.append(np.argmax(y_pred))
        y_true_list.append(np.argmax(y_test[i]))

    # cm = confusion_matrix(y_true_list, y_preditction_list)
    class_names = dataset_class_names[args.dataset]
    test_accuracy = test_correct_predictions / len(x_test)
    print("test_accuracy: ", test_accuracy)
    my_dict = {'y_prob_list':y_prob_list, 'y_preditction_list':y_preditction_list, 'y_true_list':y_true_list}



    # Log the confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=y_preditction_list,
            y_true=y_true_list,
            class_names=class_names
        )
    })
    wandb.log({
        "confusion_matrix_prob": wandb.plot.confusion_matrix(
            probs=y_prob_list,
            y_true=y_true_list,
            class_names=class_names
        )
    })

    # with open('my_dict.json', 'w') as f:
    #     json.dump(my_dict, f, indent=4)

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
            # return -np.sum(y_true * np.log(y_pred))
            epsilon = 1e-10  
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]  
        
    def forward(self, z):
        self.preactivations = []
        self.activations = [z]
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
            self.gradient_preactivation = 2 * (y_pred - y_true) / len(y_true)

        self.gradients_w[-1] = np.outer(self.activations[-2], self.gradient_preactivation)
        self.gradients_b[-1] = self.gradient_preactivation

        for i in range(len(self.weights) - 1, 0, -1):
            self.gradient_activation = np.dot(self.gradient_preactivation, self.weights[i].T)
            
            # Calculate gradient for current layer's pre-activation
            if self.args.activation == "sigmoid":
                self.gradient_preactivation = self.gradient_activation * self.gradient_sigmoid(self.preactivations[i-1])
            elif self.args.activation == "tanh":
                self.gradient_preactivation = self.gradient_activation * self.gradient_tanh(self.preactivations[i-1])
            elif self.args.activation == "ReLU":
                self.gradient_preactivation = self.gradient_activation * self.gradient_ReLU(self.preactivations[i-1])
            elif self.args.activation == "identity":
                self.gradient_preactivation = self.gradient_activation
            
            
            self.gradients_w[i-1] = np.outer(self.activations[i-1], self.gradient_preactivation)
            self.gradients_b[i-1] = self.gradient_preactivation
        # return self.gradients_w, self.gradients_b




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="Assignment1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="harshayelchuri-indian-institute-of-technology-madras")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, default=128)
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh")
    args = parser.parse_args()

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(args)
    train(args, x_train, y_train, x_val, y_val, x_test, y_test)


    # log_sample_images(x_train, y_train, args.dataset)

    
    
    wandb.finish()




if __name__ == "__main__":
    main()