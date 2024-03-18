from keras.datasets import fashion_mnist
import wandb
from sklearn.model_selection import train_test_split
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.objective_function import ObjectiveFunction
from ann.optimizer import Optimizer

def train(config, show_log): 
    train_loss_hist = []
    train_accuracy_hist = []
    val_loss_hist = []
    val_accuracy_hist = []
    
    nn = NeuralNetwork(input_size=config['input_size'], 
                       output_size=config['output_size'], 
                       hidden_layers=config['hidden_layers'], 
                       neurons=config['neurons'],  
                       activation=config['activation'], 
                       output_activation=config['output_activation'],
                       criterion=config['criterion'],
                       weight_initialization=config['weight_initialization'])
    
    optimizer = Optimizer(neural_net=nn,
                    lr=config['learning_rate'],
                    optimizer=config['optimizer'],
                    beta=config['beta'],
                    epsilon=config['epsilon'],
                    beta1=config['beta1'],
                    beta2=config['beta2'],
                    decay=config['decay'])
    
    batch_size = config['batch_size']
    c = ObjectiveFunction()
    
    for epoch in range(config['epochs']):
        for batch in range(0, X_train.shape[0], batch_size):
            # Get the batch of data
            X_batch = X_train[batch:batch+batch_size]
            Y_batch = Y_train[batch:batch+batch_size]

            Y_hat_batch = nn.forward(X_batch)
            weights, biases = nn.backward(Y_batch, Y_hat_batch)
            optimizer.step(weights, biases)
        
        optimizer.timestep += 1
        
        # Training
        Y_hat_train = nn.forward(X_train)
        train_loss = c.criterion(config['criterion'], Y_train, Y_hat_train)
        train_accuracy = np.sum(np.argmax(Y_hat_train, axis=1) == np.argmax(Y_train, axis=1)) / Y_train.shape[0]
            
        # Validation
        Y_hat_val = nn.forward(X_val)
        val_loss = c.criterion(config['criterion'], Y_val, Y_hat_val)
        val_accuracy = np.sum(np.argmax(Y_hat_val, axis=1) == np.argmax(Y_val, axis=1)) / Y_val.shape[0]
        
        if show_log:
            print("Epoch {} Train Loss {} Train Accuracy {} Val Loss {} Val Accuracy {}".format(epoch, train_loss, train_accuracy, val_loss, val_accuracy))
   
        train_loss_hist.append(train_loss)
        train_accuracy_hist.append(train_accuracy)
        val_loss_hist.append(val_loss)
        val_accuracy_hist.append(val_accuracy)
    
    return nn, train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist

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
    'optimizer': "nadam",
    'criterion': "cel",
    'decay': 0.0005,
    'weight_initialization': "random",
    'batch_size': 64,
}

wandb.init(project="cs6910-assignment-1")
wandb.config.update(network_config)

# Class Names
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load Input Data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Flatten the images
train_images = train_images.reshape(train_images.shape[0], 784) / 255
test_images = test_images.reshape(test_images.shape[0], 784) / 255

# Encode the labels
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

# Prepare data for training and validation
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1, shuffle=True, random_state=27)

nn = train(network_config)

y_pred_train = nn.forward(X_train)
y_pred_test = nn.forward(test_images)

y_true_train = np.argmax(Y_train, axis=1)
y_true_test = np.argmax(test_labels, axis=1)
preds_train = np.argmax(y_pred_train, axis=1)
preds_test = np.argmax(y_pred_test, axis=1)

# Confusion Matrix Plot
wandb.log({'confusion_matrix_train': wandb.plot.confusion_matrix(probs=None, y_true=y_true_train, preds=preds_train, class_names=CLASS_NAMES)})
wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true_test, preds=preds_test, class_names=CLASS_NAMES)})

wandb.log({'confusion_matrix_sklearn': wandb.sklearn.plot_confusion_matrix(y_true_test, preds_test, CLASS_NAMES)})

wandb.finish()