# Improving Variational Autoencoders

## COMP 551 [project 4](https://cs.mcgill.ca/~wlh/comp551/files/miniproject4_spec.pdf).

In this project we aim to improve [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf).


Variational autoencoders (VAE), a generative modelling approach to generate new data, is employed to learn the latent representation in a lower dimensional space. In this re- port, our aim is to understand, implement and develop improvements for VAE. We propose and explore different approaches to modify the baseline. First, we implemented the base- line with the original VAE model and then proposed six different extensions to check the improvements. Our results show that Denoising VAE, Conditional VAE and VAE with a CNN have the best performance in learning the true distribution, but Importance Weighted autoencoder is capable to create sharper examples.

## Install
```bash
conda create --name env --file requirements.txt
```

## Running

You can reproduce any experiment by executing its script associated.

```bash
python src/scripts/scriptX.py
```

## Project

The folder `src` contains the source code for the project. Each experiment can be replicated using the script associated with the experiment.

- `experiments`: contains the workflow for the experiment carried out.
- `models`: model used for the respective experiment.
- `scripts`: script to execute certain experiment.

## Experiments

### Experiment 1
Replication of VAE using MNIST as in the original paper.

- Encoder: Gaussian
- Decoder: Bernoulli

### Experiment 1.1
Original VAE tuned parameters.

- Encoder: Gaussian
- Decoder: Bernoulli

### Experiment 2
Replication of VAE using Frey Faces as in the original paper.

- Encoder: Gaussian
- Decoder: Gaussian

gradient clipping, change optimizer, nesterov, momentum

### Experiment 3
Improvement replacing the FNN with a CNN.

- Encoder: Gaussian
- Decoder: Bernoulli

### Experiment 4
Improvement using IWAE to tight the bound

- Encoder: Gaussian
- Decoder: Bernoulli

### Experiment 5
Implementation of Adversarial Variational Bayes, code taken from this [repo](https://github.com/wiseodd/generative-models), not included in the report.

### Experiment 6
Denoising variational autoencoders. Xavier uniform for the initialization of weights hurts sightly the model.

- Encoder: Gaussian
- Decoder: Bernoulli

### Experiment 7
Conditional VAE

- Encoder: Gaussian
- Decoder: Bernoulli

### Experiment 8
Beta VAE

## Problems

1. **Check gaussian loss for gaussian (exp 2):** For now, limiting the output of the variance in the decoder with a `tanh`, but how this affect the regularization since it would be the same as having a normal distribution _N(u(x), I)_. Also, it changes the value of the bound used to compare results.
2. **Check results for IWAE**, results seem a bit tricky. It might because we're getting zero-probabilities, and in those cases we assign zero, ie log 0 = 0.


## Results

See the report for results.
