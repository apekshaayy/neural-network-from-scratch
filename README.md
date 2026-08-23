# Neural Network from Scratch: MNIST Digit Classification

A 2-layer neural network built entirely with NumPy trained to classify handwritten digits from the MNIST dataset. Neither Tensorflow nor PyTorch was used in this project.

## Overview

This project implements forward propagation, backpropagation, and gradient descent by hand to understand what's actually happening under the hood of a neural network. It trains on the classic MNIST dataset (784 input pixels → hidden layer → 10 output classes).

**Final accuracy:** `~82.19%` on the training set, `84.29%` on the test set.

## Architecture

- **Input layer:** 784 units (28x28 flattened pixel values, normalized to [0, 1])
- **Hidden layer:** 10 units, ReLU activation
- **Output layer:** 10 units, Softmax activation (digits 0–9)
- **Training:** Batch gradient descent with backpropagation

## Project Structure

```
neural-network-from-scratch/
├── neural_network_from_scratch_mnist.ipynb   # MNIST
└── README.md
```

## Setup

```bash
git clone https://github.com/apekshaayy/neural-network-from-scratch.git
cd neural-network-from-scratch
jupyter notebook neural_network_from_scratch_mnist.ipynb
```

### Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn (for `fetch_openml`)

## Notebook Walkthrough

1. **Data Loading** — Fetch MNIST via `sklearn.datasets.fetch_openml`
2. **Preprocessing** — Train/test split, pixel normalization, label formatting
3. **Model** — `init_params`, `forward_propogation`, `backward_propogation`, `update_params`
4. **Training** — `gradient_descent` loop with accuracy logging
5. **Evaluation** — Test set accuracy + visual predictions grid

## Sample Predictions

*<img width="1742" height="631" alt="image" src="https://github.com/user-attachments/assets/a1446c37-f2c5-413c-88dc-6aee7e397091" />*

## What I Learned / Debugged

- Weight initialization matters a lot — `np.random.randn() - 0.5` (unscaled normal distribution) caused training to stall near random-guess accuracy; switching to `np.random.rand() - 0.5` (small uniform init) fixed it.
- Normalizing pixel values to [0, 1] was essential for stable gradients.
- Built utilities to visualize predictions and sanity-check gradients when accuracy plateaued.

## Acknowledgements

This project was built while following [Samson Zhang's "Building a Neural Network from Scratch"](https://www.youtube.com/watch?v=w8yWXqWQYmU) video as a guide and reference. I implemented, debugged, and extended the code myself — including fixing an initialization bug that was capping training accuracy near random-guess levels, and adding visualization/evaluation utilities.
