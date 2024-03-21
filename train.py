import argparse
import numpy as np
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split

from src.ann.neural_network import NeuralNetwork
from src.ann.optimizer import Optimizer
from src.ann.objective_functions import ObjectiveFunction

WANDB_PROJECT = "CS6910_AS1"
WANDB_ENTITY = "ed23s037"

network_config = {
    'wandb_project': WANDB_PROJECT,
    'wandb_entity': WANDB_ENTITY,
    'dataset': 'fashion_mnist',
    'epochs': 20,
    'batch_size': 64,
    'criterion': 'cross_entropy',
    'optimizer': 'adam',
    'learning_rate': 0.005,
    'momentum': 0.8,
    'beta': 0.9,
    'beta1': 0.9,
    'beta2':0.9999,
    'epsilon': 1e-8,
    'weight_decay': 0.0005,
    'weight_init': "xavier",
    'num_layers': 5,
    'hidden_size':128,
    'activation':'relu',
    'input_size': 784,
    'output_size': 10,
    'output_activation':'softmax',
}

parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--wandb_project", type=str, default=WANDB_PROJECT, help="Wandb project name", required=True)
parser.add_argument("-we", "--wandb_entity", type=str, default=WANDB_ENTITY, help="Wandb entity name", required=True)
parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", help="Dataset to use choices=['fashion_mnist', 'mnist']")
parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("-l", "--loss", type=str, default="cross_entropy", help="Loss function to use choices=['cross_entropy', 'mean_squared_error']")
parser.add_argument("-o", "--optimizer", type=str, default="adam", help="Optimizer to use choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.005, help="Learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.8, help="Momentum for Momentum and NAG")
parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for RMSProp")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam and Nadam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for Adam and Nadam")
parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for Adam and Nadam")
parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay")
parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", help="Weight initialization choices=['random', 'xavier']")
parser.add_argument("-nhl", "--num_layers", type=int, default=5, help="Number of hidden layers")
parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Hidden size")
parser.add_argument("-a", "--activation", type=str, default="relu", help="Activation function choices=['sigmoid', 'tanh', 'relu']")

def get_dataset(dataset):
    if dataset == 'mnist':
        return mnist.load_data()
    return fashion_mnist.load_data()

def train(config):
    # Load Input Data
    (train_images, train_labels), (test_images, test_labels) = get_dataset(config['dataset'])

    # Flatten the images
    train_images = train_images.reshape(train_images.shape[0], 784) / 255
    X_test = test_images.reshape(test_images.shape[0], 784) / 255

    # Encode the labels
    train_labels = np.eye(10)[train_labels]
    Y_test = np.eye(10)[test_labels]

    # Prepare data for training and validation
    X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1, shuffle=True, random_state=27)
    
    train_loss_hist = []
    train_accuracy_hist = []
    val_loss_hist = []
    val_accuracy_hist = []
    
    nn = NeuralNetwork(config)
    optimizer = Optimizer(nn=nn, config=config)
    
    batch_size = config['batch_size']
    criterion = ObjectiveFunction(method = config['criterion'])
    
    for epoch in range(config['epochs']):
        for batch in range(0, X_train.shape[0], batch_size):
            # Get the batch of data
            X_batch = X_train[batch:batch+batch_size]
            Y_batch = Y_train[batch:batch+batch_size]

            Y_hat_batch = nn.forward(X_batch)
            nn.backward(X_batch, Y_batch, Y_hat_batch)
            optimizer.step()
        
        optimizer.timestep += 1
        
        # Training
        Y_hat_train = nn.forward(X_train)
        train_loss = criterion.get_loss(Y_train, Y_hat_train)
        train_accuracy = np.sum(np.argmax(Y_hat_train, axis=1) == np.argmax(Y_train, axis=1)) / Y_train.shape[0]
            
        # Validation
        Y_hat_val = nn.forward(X_val)
        val_loss = criterion.get_loss(Y_val, Y_hat_val)
        val_accuracy = np.sum(np.argmax(Y_hat_val, axis=1) == np.argmax(Y_val, axis=1)) / Y_val.shape[0]
        
        print("Epoch {} Train Loss {} Train Accuracy {} Val Loss {} Val Accuracy {}".format(epoch, train_loss, train_accuracy, val_loss, val_accuracy))
   
        train_loss_hist.append(train_loss)
        train_accuracy_hist.append(train_accuracy)
        val_loss_hist.append(val_loss)
        val_accuracy_hist.append(val_accuracy)
        
    return nn, train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist


args = parser.parse_args()
network_config.update(vars(args))

# Print the parameters
print("Parameters:")
for key, value in network_config.items():
    print(f"{key}: {value}")

train(network_config)