# Self_Supervised_CIFAR
Compare the performance of supervised learning and self-supervised learning in CIFAR-10 or CIFAR-100 image classification tasks


## Setup
Requirements: torch, torchvision, torchnet

Run:
`mkdir datasets`

`mkdir experiments`


## Self-supervised Pre-training

The project referred the self-supervised learning strategy from **Unsupervised Representation Learning by Predicting Image Rotations**.

Train ResNet18 on CIFAR-10 with `python main.py --exp=CIFAR10_RotNet_ResNet18`

## Linear Classification Protocol
Refer from **Momentum Contrast for Unsupervised Visual Representation Learning**

`python main.py --exp=CIFAR100_LinearClassifiers_CIFAR10_RotNet_ResNet18_Features`

## Baseline
The baseline is ResNet18 trained on CIFAR-10. You can train via `python train.py` in `baseline` folder.