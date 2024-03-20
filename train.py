import argparse

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
    'num_layers': 1,
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
parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("-l", "--loss", type=str, default="cross_entropy", help="Loss function to use choices=['cross_entropy', 'mean_squared_error']")
parser.add_argument("-o", "--optimizer", type=str, default="sgd", help="Optimizer to use choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for Momentum and NAG")
parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSProp")
parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for Adam and Nadam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for Adam and Nadam")
parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for Adam and Nadam")
parser.add_argument("-w_d", "--weight_decay", type=float, default=.0, help="Weight decay")
parser.add_argument("-w_i", "--weight_init", type=str, default="random", help="Weight initialization choices=['random', 'xavier']")
parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Hidden size")
parser.add_argument("-a", "--activation", type=str, default="sigmoid", help="Activation function choices=['sigmoid', 'tanh', 'relu']")

args = parser.parse_args()
network_config.update(vars(args))

# Print the parameters
print("Parameters:")
for key, value in network_config.items():
    print(f"{key}: {value}")

def get_dataset():
    pass

def train():
    pass