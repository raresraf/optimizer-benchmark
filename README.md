Benchmark Suite for Stochastic Gradient Descent Optimization Algorithms in Pytorch
-----

Credits https://github.com/ifeherva/optimizer-benchmark for the framework.

This repository contains code to benchmark novel stochastic gradient descent algorithms on
the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

If you want your algorithm to be included open an issue here.

Requirements: Python 3.6+, Pytorch 1.3+, tqdm

Supported optimizers:

1. Stochastic Gradient Descent with Momentum (SGDM)
1. Adam: A method for stochastic optimization (ADAM) [[arXiv]](https://arxiv.org/abs/1412.6980)
1. RAdam: On the Variance of the Adaptive Learning Rate and Beyond [[arXiv]](https://arxiv.org/abs/1908.03265)
1. AdaBound: Adaptive Gradient Methods with Dynamic Bound of Learning
   Rate [[ICLR2019]](https://openreview.net/pdf?id=Bkg3g2R9FX)

Results:
------

![](assets/test_accuracies.png)
