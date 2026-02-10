# PV021 Project -- Deep Learning from Scratch in Java

## Members
- **Joan Bejarano Montero**
- **Evan Aubriet**

## Project Overview
This repository contains a **purely "from scratch" implementation** of a Deep Neural Network in Java, developed for the PV021 course. The goal is to solve the **Fashion-MNIST** classification problem without using any third-party machine learning libraries (like PyTorch, TensorFlow, or Deeplearning4j) or high-level matrix libraries.

Every component of the deep learning pipeline has been manually implemented to ensure maximum performance and understanding:
- **Matrix Mathematics:** Custom implementation of matrix multiplication, transposition, and vector operations.
- **Forward Propagation:** Layer-by-layer activation handling.
- **Backpropagation:** Manual calculation of gradients (deltas) for weights and biases using the chain rule.
- **Optimization:** Mini-batch Gradient Descent with Momentum.

## Technical Architecture
Based on the implementation in `ProjectNetwork.java` and `Layer.java`, the model uses the following specifications:

### Network Topology (Fashion MNIST)
The network is a **Feed-Forward Neural Network (Multi-Layer Perceptron)** with the following structure:
1.  **Input Layer:** 784 neurons (corresponding to 28x28 pixel images).
2.  **Hidden Layer 1:** 128 neurons, **ReLU** activation.
3.  **Hidden Layer 2:** 64 neurons, **ReLU** activation.
4.  **Output Layer:** 10 neurons, **Softmax** activation (representing the 10 fashion classes).

### Hyperparameters & Optimization
- **Initialization:** Xavier Initialization (to maintain variance across layers).
- **Optimizer:** Mini-batch Stochastic Gradient Descent (SGD).
    - **Batch Size:** 64.
    - **Momentum:** 0.9 (implemented via velocity vectors in `Layer.java`).
- **Learning Rate Strategy:**
    - Initial Learning Rate: `0.015`.
    - **Step Decay:** The learning rate is halved at epochs 10, 13, 16, and 19 to fine-tune convergence.
- **Epochs:** 20.

## Project Structure
```text
.
├── data/                        # Dataset directory (See Setup)
├── src/
│   ├── datatreatments/          # Data loading and parsing (FashionMNISTDataLoader)
│   ├── neuralstypes/            # Core Neural Network components (Layer, Network)
│   │   └── networks/            # Specific architectures (ProjectNetwork, XORNetwork)
│   ├── utility/                 # Math helpers (MatrixMath) and Activation functions
│   └── Main.java                # Entry point and menu logic
├── run.sh                       # Compilation and execution script
├── evaluate.sh                  # Script to evaluate accuracy
└── README.md
```

## Usage
### 1. Execution
To compile and run the project, simply use the provided shell script:
```./run.sh```

**Interactive vs. Automatic Mode**: The behavior of the application depends on the ```boolean input``` flag in ```Main.java```:
- Automatic Mode (Default): If ```input = false```, the system will automatically run the ProjectNetwork (Fashion MNIST) using a random seed. This is the default configuration for submission.
- Interactive Mode: If you modify the code to ```input = true```, the program will launch a console menu allowing you to:
  1. Run XOR Network: A benchmark 2-2-1 network to test backpropagation logic.
  2. Run Fashion MNIST: The main project network. You will be prompted to enter a specific seed manually.

### 2. Output
Upon successful execution, the program will generate two files in the root directory:
- ```train_predictions.csv```: Predictions for the training set.
- ```test_predictions.csv```: Predictions for the test set.

### 3. Evaluation
To check the accuracy of the generated predictions against the ground truth:

```./evaluate.sh```

Note on Server Deployment (AURA)
If you need to deploy this project to the university AURA server, a script named ```upload_data_aura.sh``` is included to facilitate the transfer of data and source code.