# MNIST Digit Classification with PyTorch

This project is all about getting a neural network to recognize handwritten digits from the famous MNIST dataset. I've built it with PyTorch, keeping things simple yet remarkably effective. It includes clean, modular scripts for training and then putting that trained model to work. Plus, it's smart enough to train itself if I haven't got the model ready to go.

## Project Structure

Here's how I've organized things:

- ``train.py``: This script is where the primary work happens. It trains my neural network using the MNIST training data and then saves the fully trained model as ``mnist.pth``.

- ``main.py``: This is your go-to for making predictions. It loads up my trained ``mnist.pth`` model and then tests it out on some random digits from the test set. If it can't find mnist.pth, it'll automatically kick off ``train.py`` to get things ready for you.

- ``mnist.pth``: This file contains all the learned information from my training, making it the core of my digit classification.

## What's Inside (Features)

I've packed in some thoughtful features to make this project effective:

- Simple Network, Powerful Results: I'm using a straightforward fully connected neural network. To enhance performance, I've integrated Batch Normalization and a key choice for the activation function: ``Mish``. While ``Mish`` can be a bit more computationally intensive than alternatives like ``ReLU`` or ``Leaky ReLU``, I found that it consistently outperforms them in terms of model accuracy and learning capabilities for this task.

- Improved Robustness: I am making my model more resilient by applying **random rotations** to the training data. This helps it better recognize digits even if they're slightly skewed.

- Efficient Training: I've included a learning rate scheduler to help the training process converge smoothly and efficiently. For my loss function, I'm using ``CrossEntropyLoss``. A neat thing about ``CrossEntropyLoss`` in PyTorch is that it intelligently combines the ``LogSoftmax`` and ``Negative Log Likelihood loss`` in one go, so you don't have to explicitly add a ``Softmax`` layer in your network's final output. It handles that part for you, which simplifies the code and often improves numerical stability.

- Automatic Training Fallback: If you run ``main.py`` and the ``mnist.pth`` model isn't present, the script will automatically initiate the training process for you.

- GPU Support: If you have a ``CUDA-enabled GPU``, my setup can leverage it for significantly faster training and inference.

## What You'll Need

To get this up and running, just make sure you have:

- Python 3.11 (or a more recent version, though I'm building this with 3.11 in mind)

- PyTorch
- torchvision

You can get all these dependencies installed quite easily by running this command:

```bash
pip install torch torchvision
```
