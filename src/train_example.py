# Load all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

from ann.neural_network import NeuralNetwork
from ann.optimizer import Optimizer
from ann.objective_functions import ObjectiveFunction

def train(config):
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

# Load Input Data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Flatten the images
train_images = train_images.reshape(train_images.shape[0], 784) / 255
X_test = test_images.reshape(test_images.shape[0], 784) / 255

# Encode the labels
train_labels = np.eye(10)[train_labels]
Y_test = np.eye(10)[test_labels]

# Prepare data for training and validation
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1, shuffle=True, random_state=27)

network_config = {
    'input_size': 784,
    'output_size': 10,
    'hidden_layers': 1,
    'neurons':256,
    'activation':'sigmoid',
    'output_activation':'softmax',
    'learning_rate': 0.005,
    'beta': 0.8,
    'beta1': 0.9,
    'beta2':0.9999,
    'epsilon': 1e-8,
    'epochs': 10,
    'optimizer': "sgd",
    'criterion': "cel",
    'decay': 0.0005,
    'weight_initialization': "random",
    'batch_size': 64,
}
        
nn, tl, ta, vl, va = train(network_config)