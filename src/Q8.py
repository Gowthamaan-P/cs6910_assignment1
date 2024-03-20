from keras.datasets import fashion_mnist
import wandb
from sklearn.model_selection import train_test_split
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.objective_functions import ObjectiveFunction
from ann.optimizer import Optimizer

sweep_config = {
    'method': 'random',
    'name': 'Q4_SWEEP',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize',
    },
    'parameters': {
        'input_size': {
            'value': 784
        },
        'output_size': {
            'value': 10
        },
        'hidden_layers': {
            'values': [3, 4, 5]
        },
        'neurons': {
            'values': [32, 64, 128]
        },
        'activation': {
            'values': ['sigmoid', 'tanh', 'relu',]
        },
        'output_activation': {
            'value': 'softmax'
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'decay': {
            'values': [0, 0.0005, 0.000005]
        },
        'epochs': {
            'value': [5, 10]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_initialization': {
            'values': ['xavier', 'random']
        },
        'beta': {
            'value': [0.7, 0.8, 0.9]
        },
        'beta1':{
            'value': 0.9
        },
        'beta2':{
            'value': 0.9999
        },
        'epsilon': {
            'value': 1e-8
        },
        'criterion': {
            'value': 'mse'
        },
    }
}

def accuracy(y, y_hat):
    return np.sum(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)) / y.shape[0]
    

def wandb_sweep():
    train_loss_hist = []
    train_accuracy_hist = []
    val_loss_hist = []
    val_accuracy_hist = []

    run = wandb.init()
    config = wandb.config
    run.name = f"hl_{config['hidden_layers']}_nu_{config['neurons']}_ac_{config['activation']}_lr_{config['learning_rate']}_bs_{config['batch_size']}_opt_{config['optimizer']}_de_{config['decay']}_init_{config['weight_initialization']}"


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

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        train_loss_hist.append(train_loss)
        train_accuracy_hist.append(train_accuracy)
        val_loss_hist.append(val_loss)
        val_accuracy_hist.append(val_accuracy)

    # Testing
    Y_hat_test = nn.forward(X_test)
    test_loss = criterion.get_loss(Y_test, Y_hat_test)
    test_accuracy = accuracy(Y_test, Y_hat_test)
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

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

# Setup Wandb
wandb.login(key='5da0c161a9c9720f15195bb6e9f05e44c45112d1')
wandb.init(project="CS6910_AS1", entity='ed23s037')

# Do Sweep
wandb_id = wandb.sweep(sweep_config, project="CS6910_AS1")
wandb.agent(wandb_id, function=wandb_sweep, count=100)

# Finish
wandb.finish()