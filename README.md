# History Vault — Practical ML & DL Playground


This repository is a living history vault of things I’ve learned by building practical projects end-to-end. It’s organized as a sequence of experiments and small projects that progressively move from simple classifiers to full-scale systems like object detectors, transformers, and retrieval-augmented-generation (RAG) pipelines.

This is a hands-on learning log — not just notes. Every entry is code-first: I build working models, evaluate them, and iterate. The goal is to gain the intuition and engineering skills needed to eventually build improved versions of GPT-2 and YOLOv8.

## High-level roadmap
1. Single-class classifiers 
- Start simple: MLPs on MNIST / Fashion-MNIST.
- Learn dataset handling, training loops, optimizers, schedulers, and regularization.
- Tools used: PyTorch, torchvision, NumPy.
2. Muti Object detection 
- Synthetic multi-object datasets from MNIST / Fashion-MNIST to learn bounding-box regression and NMS.
- Implement detection heads, multi-task losses (classification + bbox), IoU matching, and non-maximum suppression.
- Prototype a tiny YOLO-like detector from scratch.
3. Convolutional backbones & scale
- Replace MLPs with CNNs to learn spatial features and receptive field concepts.
- Add batch-norm, residual/bottleneck blocks, and training best-practices (data augmentation, mixed precision).
4. Transformers and sequence models
- Build a transformer from first principles (attention, positional encodings, LM head).
- Train small language models (toy GPT-2 variant) to understand scaling and optimization.
5. RAG pipelines and end-to-end agents

- Build retrieval + generation systems (ingest, vector index, retriever, generator, API/server).

- Integrate with lightweight UIs for demos.

## What’s in this repo today
The repository contains a snapshot of experiments and working scripts. Notable files/folders:
-``Numerical-MNIST/`` — MNIST training scripts and saved model checkpoints.