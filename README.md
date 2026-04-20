
# Neural Network from Scratch

A modular, step-by-step implementation of core neural network components built entirely from scratch using Python and NumPy. This repository breaks down the mathematics behind deep learning into clean understandable, and executable scripts without relying on heavy machine learning frameworks like TensorFlow or PyTorch.

## Core Architecture Features

The project is organised into modular directories to isolate and demonstrate indivitual concepts:

### 1. Activation Functions  (`/activation_functions`)
Contains mathematical implementations and visualization scripts for common neural network activation functions:
* *ReLU & Leaky ReLU*
* *Sigmoid*
* *Tanh*
* *Softmax*

### 2. Forward Propagation  (`/forward_propagation`)
Demonstrates how input data flows through the layers of a network to generate predictions:
* Basic feedforward network implementation.
* Practical application and testing on the Iris dataset.

### 3. Backward Propagation  (`/backward_propagation`)
Explores the learning mechanisms and optimization algorithms:
* Gradient Descent implementation.
* Linear Regression conceptual breakdown.
* Toy datasets for testing mathematical weight updates.
<br>
<br>
  <img width="1200" height="600" alt="Figure_3" src="https://github.com/user-attachments/assets/6363c123-8dd9-4d1e-a9bd-bbba6c76f5a4" />
<br>
<br>

## Tech Stack

* *Python*
* *NumPy* (for efficient matrix operations and linear algebra)
* Matplotlib

## Installation & Prerequisites

To explore and run these scripts, you will need Python installed along with the required numerical libraries.

```bash
pip install numpy
```

## How to Run

You can run the main entry point of the project:
```bash
python main.py
```

Alternatively, you can execute any of the specific conecpt scripts indivitually to see how they work. For example, to visualize the activation functions:
```bash
python activation_functions/
ex_vis_act_fn.py
```
