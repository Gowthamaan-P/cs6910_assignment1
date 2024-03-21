<div style="text-align: justify;">

# CS6910 Fundamentals of Deep Learning - Assignment 1

This repository contains all files for the first assignment of the CS6910 - Fundamentals of Deep Learning course at IIT Madras.

## Contents

* [Task](#task)
* [Submission](#submission)
* [Dataset](#dataset)
* [Implementation Details](#implementation-details)
  - [Structure](#structure)
  - [Optimizers](#optimizers)
  - [Criterion](#criterion)
  - [Backpropagation](#backpropagation)
  - [Flexibility](#flexibility)
* [Tools and Libraries Used](#tools-and-libraries-used)
  - [Packages Used](#packages-used)
  - [Installation](#installation)
* [Usage](#usage)
  - [Running Manually](#running-manuallyn)
  - [Running a Sweep using Wandb](#running-a-sweep-using-wandb)
  - [Customization](#customization)

## Task

The task involves implementing a FeedForward Neural Network with Backpropagation from scratch.

## Submission

My WandB project: https://wandb.ai/ed23s037/CS6910_AS1
My WandB report: https://api.wandb.ai/links/ed23s037/nfa8vpme

## Dataset

The dataset utilized for this assignment comprises Fashion MNIST and MNIST, both available in the `keras.datasets` module and can be imported directly into the 
```python
from keras.datasets import fashion_mnist
from keras.datasets import mnist
```

## Implementation Details

In this section, we delve into the implementation of the feedforward network and backpropagation, detailing the structure and functionalities of each component.

### Structure

The implementation comprises four main files located under the `src/ann` directory:

1. **neural_layer.py**: This file describes a single layer in the neural network. It encompasses parameters such as weights, biases, activation functions, gradients, and more.
   
2. **neural_network.py**: Responsible for defining the entire neural network based on the provided configuration. It integrates neural layers into the network and facilitates forward and backward passes.
   
3. **optimizer.py**: Defines various optimizers utilized for gradient descent. Optimization is performed on a per-layer basis.
   
4. **objective_functions.py**: This file contains definitions for loss functions and their derivatives required for backpropagation.

### Optimizers

The implemented optimizers offer a range of options to facilitate efficient gradient descent:

- **SGD (Stochastic Gradient Descent)**
- **Momentum (Momentum SGD)**
- **NAG (Nesterov Accelerated Gradient - optimized version)**
- **RMSProp (Root Mean Square Propagation)**
- **Adam (Adaptive Moment Estimation)**
- **Nadam (Nesterov Adaptive Moment Estimation)**

### Criterion

Implemented loss functions cater to various optimization needs:

- **Cross Entropy**
- **Mean Squared Error**

### Backpropagation

Backpropagation is implemented iteratively, computing delta and error for each layer. Gradients are then computed by multiplying the delta with inputs.

### Flexibility

The implementation offers high flexibility, allowing for customization of various network aspects, including:

- Different activation functions per layer
- Sparse connections
- Varying numbers of neurons
- Input batch size
- Output activation functions

With minimal modifications, a highly customizable neural network can be tailored to specific requirements. This structure and flexibility empower users to experiment with diverse network configurations and optimization strategies.


## Tools and Libraries Used

The implementation of the project utilizes the following tools and libraries:

- **Python 3.10.1**: The core programming language used for implementing all aspects of the project.

- **WandB (Weights and Biases)**: WandB is utilized for running experiments, hyperparameter tuning, visualization, and more.

### Packages Used

1. **Numpy 1.21.0**: Used to implement the neural layer and perform computations

2. **Matplotlib 3.4.2**: Used for plotting graphs, histograms, and other statistical visualizations.

3. **Keras 2.7.0**: Used for loading datasets.

4. **WandB 0.16.4**

5. **Scikit-learn 0.24.2**: Used for plotting confusion matrices and splitting data.

### Installation

All the above packages are listed in the `requirements.txt` file. To install them, simply execute the following command:

```sh
$ pip install -r requirements.txt
```

By installing the listed packages, you can ensure that all necessary dependencies are met for running the project smoothly.

## Usage

### Running Manually

To execute the file manually, use the following command:

```sh
$ python3 train.py -wp <wandb_project_name> -we <wandb_entity_name>
```

You can also modify the following list of available options along with brief information about each:

```sh
$ python3 train.py -h
```

#### Options

| Option                           | Description                                                                                 |
|----------------------------------|---------------------------------------------------------------------------------------------|
| `-h, --help`                     | Display help message and exit                                                              |
| `-wp WANDB_PROJECT`              | Wandb project name                                                                          |
| `-we WANDB_ENTITY`               | Wandb entity name                                                                           |
| `-d DATASET`                     | Dataset to use (`fashion_mnist` or `mnist`)                                                 |
| `-e EPOCHS`                      | Number of epochs                                                                            |
| `-b BATCH_SIZE`                  | Batch size                                                                                  |
| `-l LOSS`                        | Loss function (`cross_entropy` or `mean_squared_error`)                                      |
| `-o OPTIMIZER`                   | Optimizer to use (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`)                     |
| `-lr LEARNING_RATE`              | Learning rate                                                                               |
| `-m MOMENTUM`                    | Momentum for Momentum and NAG                                                               |
| `-beta BETA`                     | Beta for RMSProp                                                                            |
| `-beta1 BETA1`                   | Beta1 for Adam and Nadam                                                                    |
| `-beta2 BETA2`                   | Beta2 for Adam and Nadam                                                                    |
| `-eps EPSILON`                   | Epsilon for Adam and Nadam                                                                  |
| `-w_d WEIGHT_DECAY`              | Weight decay                                                                                |
| `-w_i WEIGHT_INIT`               | Weight initialization (`random` or `xavier`)                                                |
| `-nhl NUM_LAYERS`                | Number of hidden layers                                                                     |
| `-sz HIDDEN_SIZE`                | Hidden size                                                                                 |
| `-a ACTIVATION`                  | Activation function (`sigmoid`, `tanh`, `relu`)                                              |

These options provide a comprehensive set of configurations to customize the neural network model and the training process according to specific requirements and experimentation needs.

### Running a Sweep using Wandb

To conduct a sweep using wandb, set the values of count and project name in `src/Q4.py`, then execute the following command:

```sh
$ python3 src/Q4.py
```

### Customization
Refer to the `src/train_example.py` for training code. The `notebooks` folder has all the jupyter notebooks enlisting the different trails and base code for this implmentation. 
</div>