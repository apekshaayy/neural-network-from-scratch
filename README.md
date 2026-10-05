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


# Test-Time Learning: Function Extrapolation with LoRA

A PyTorch implementation of **Test-Time Learning (TLM)** — adapting a pretrained network on unlabeled test inputs using a self-supervised consistency signal and LoRA — applied to simple mathematical functions (sine, linear) as a toy testbed for the underlying algorithm from a Test-Time Learning for LLMs paper.

## Overview

A small MLP is pretrained on one input range of a function (e.g. `sin(x)` on `x ∈ [-3, 3]`), then evaluated on a **shifted, out-of-distribution range** (`x ∈ [3, 5]`) where it has never seen data. Without adaptation, the network extrapolates poorly — ReLU networks settle into a fixed linear behavior outside the training range, causing predictions to flatten rather than continue the function's true shape.

This project implements **Algorithm 1** from the paper: at test time, a low-rank LoRA adapter is injected into the pretrained model and updated *online*, batch by batch, using only a self-supervised consistency loss — no ground-truth labels are used during adaptation.

## Algorithm

For each batch of test inputs:
1. **Predict** `ỹ` using the current (base + LoRA) model.
2. **Score** each input by its *consistency under perturbation* — small random noise is added to `x` several times, and the variance of the resulting predictions is used as an inverse-confidence score. Low variance ⇒ high confidence.
3. **Select** the top-`k` most confident inputs in the batch.
4. **Update** only the LoRA parameters (`A`, `B`) by minimizing the consistency loss (prediction variance) on the selected subset — the base network stays frozen throughout.

LoRA parameters persist and accumulate updates across the test stream (no reset between batches), so adaptation is online rather than per-sample.

## Architecture

- **Input layer:** 1 unit (scalar `x`)
- **Hidden layers:** 3 × Linear(64) with ReLU
- **Output layer:** 1 unit, linear (regression, no output activation)
- **LoRA injection point:** last hidden layer only — base weights frozen, trainable low-rank matrices `A (rank, in_features)` and `B (out_features, rank)` added on top, `B` zero-initialized so the adapted model is identical to the base model before any test-time updates

## Project Structure

```
neural-network-from-scratch
├── test-time-learning.inpyb
├── mnsist-digit-classifier.inpyb
└── README.md
```

## Setup

```bash
git clone https://github.com/apekshaayy/test-time-learning-functions.git
cd test-time-learning-functions
jupyter notebook ttl_sine.ipynb
```

### Dependencies
- torch
- matplotlib

## Notebook Walkthrough

1. **Data generation** — `x_train` sampled from the in-distribution range with added noise; `x_test` sampled from a shifted, noise-free range to simulate distribution shift
2. **Base model** — `SineMLP`, a 3-hidden-layer MLP trained normally via MSE loss (`pretrain`)
3. **LoRA adapter** — `LoRALinear` wraps a frozen `nn.Linear`, adding trainable low-rank `A`/`B`; `inject_lora` freezes the base model and swaps in the adapter on the last hidden layer
4. **Selection score** — `selection_score` computes per-point prediction variance under repeated small perturbations (no-grad, since it's only used for ranking)
5. **Consistency loss** — `consistency_loss` recomputes the same perturbation variance, this time with gradients enabled, as the self-supervised objective minimized during adaptation
6. **Test-time adaptation loop** — `test_time_adapt_and_predict` runs Algorithm 1 batch-by-batch over the test stream, updating only LoRA parameters
7. **Evaluation** — baseline (frozen, no adaptation) vs. TLM-adapted predictions compared against the true function via MSE and `plot_results`

## Status

Core components (model, LoRA layer, selection score, consistency loss, adaptation loop) are implemented; end-to-end debugging of the full pipeline is in progress.

## What I Learned / Debugged (so far)

- `torch.no_grad()` vs. setting `requires_grad = False` are not interchangeable — the latter permanently disables gradients on whatever parameters it touches, which silently broke LoRA training when first used inside `selection_score`.
- ReLU networks extrapolate as a fixed linear function once their active units stop changing outside the training range — this is *why* the un-adapted baseline flattens on the OOD test range, not a bug.
- Zero-initializing LoRA's `B` matrix (not `torch.empty`, which gives uninitialized memory) is essential so the adapted model starts identical to the pretrained base model.

## Acknowledgements

Algorithm design adapted from a Test-Time Learning for LLMs paper (via a research internship prospect), reimplemented here on simple 1D function regression as a tractable testbed for understanding the mechanism.
