# Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates

Code for reproducing results published in the paper ["Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates"](https://ilija139.github.io/pub/aaai-17.pdf) (AAAI-17) by Ilija Ilievski, Taimoor Akhtar, Jiashi Feng, and Christine Annette Shoemaker. 

[arXiv](https://arxiv.org/abs/1607.08316) -- [PDF](https://ilija139.github.io/pub/aaai-17.pdf) -- [Supplement](https://ilija139.github.io/pub/aaai-17-sup.pdf) -- [Poster](https://ilija139.github.io/pub/aaai-17_poster.pdf)

### About

The paper presents an algorithm for Hyperparameter Optimization via RBF and Dynamic coordinate search (HORD).
HORD searches a surrogate model of the expensive function (full training and validation of deep learning algorithm) for the most promising hyperparameter values through dynamic coordinate search and thus requires many fewer expensive function evaluations. 
HORD does well in low dimensions but it is exceptionally better in higher dimensions.
 Extensive evaluations on MNIST and CIFAR-10 for four deep learning algorithms demonstrate HORD significantly outperforms the well-established Bayesian optimization methods such as GP, SMAC, and TPE. 
For instance, on average, HORD is more than 6 times faster than GP-EI in obtaining the best configuration of 19 hyperparameters.


<img src="https://github.com/ilija139/HORD/blob/master/figures/exp-6D.jpg" width="420px"> <img src="https://github.com/ilija139/HORD/blob/master/figures/exp-6D_B.jpg" width="420px">

**Left**: Efficiency comparison of HORD with baselines. The methods are used for optimizing an MLP network with 6 hyperparameters on the MNIST dataset (**6-MLP**). We plot validation error curves of the compared methods  against  number of the function evaluations (averaged over 10 trials).

**Right**: Mean validation error v.s. number of function evaluations  of different methods for optimizing 6 hyperparameters of MLP (**6-MLP**) network on MNIST (averaged over 10 trials). One dot represents validation error of an algorithm at the corresponding evaluation instance.

<img src="https://github.com/ilija139/HORD/blob/master/figures/exp-8D.jpg" width="420px"> <img src="https://github.com/ilija139/HORD/blob/master/figures/exp-8D_B.jpg" width="420px">

**Left**: Efficiency comparison of HORD with baselines. The methods are used for optimizing a CNN with 8 hyperparameters on the MNIST dataset (**8-CNN**). We plot validation error curves of the compared methods  against  number of the function evaluations (averaged over 5 trials). 

**Right**: Mean validation error v.s. number of function evaluations of different methods for optimizing 8 hyperparameters of CNN (**8-CNN**) on MNIST (averaged over 5 trials). One dot represents validation error of an algorithm at the corresponding evaluation instance.

<img src="https://github.com/ilija139/HORD/blob/master/figures/exp-15D.jpg" width="420px"> <img src="https://github.com/ilija139/HORD/blob/master/figures/exp-15D_B.jpg" width="420px">

**Left**: Efficiency comparison of HORD and HORD-ISP with baselines. The methods are used for optimizing a CNN with 15 hyperparameters on the MNIST dataset (**15-CNN**). We plot  validation error curves of the compared methods  against  number of the function evaluations (averaged over 5 trials). HORD and HORD-ISP show to be significantly more efficient than other methods.

**Right**: Mean validation error  v.s. number of function evaluations  of different methods for optimizing 15 hyperparameters of CNN (**15-CNN**) on MNIST (averaged over 5 trials). One dot represents validation error of an algorithm at the corresponding evaluation instance. 

<img src="https://github.com/ilija139/HORD/blob/master/figures/exp-19D.jpg" width="420px"> <img src="https://github.com/ilija139/HORD/blob/master/figures/exp-19D_B.jpg" width="420px">

**Left**: Efficiency comparison of HORD and HORD-ISP with baselines for optimizing a CNN with 19 hyperparameters on the CIFAR-10 dataset (**19-CNN**). We plot validation error curves of the compared methods against number of the function evaluations (averaged over 5 trials). HORD and HORD-ISP show to be significantly more efficient than other methods. HORD-ISP only takes 54 function evaluations to achieve the lowest validation error that the best baseline (SMAC) achieves after 200 evaluations. 

**Right**: Mean validation error v.s. number of function evaluations  of different methods for optimizing 19 hyperparameters of CNN (**19-CNN**) on CIFAR-10. One dot represents validation error of an algorithm at the corresponding evaluation instance. After n<sub>0</sub> evaluations, the searching of HORD and HORD-ISP starts to focus on the hyperparameters with smaller validation error (<35%), in stark contrast with other methods.

For more details download the paper from [arxiv](https://arxiv.org/abs/1607.08316).

### Installation

The HORD algorithm presented in the paper uses the open-source pySOT toolbox by David Eriksson, David Bindel, and Christine Shoemaker.
To install pySOT go to [pySOT](https://github.com/dme65/pySOT) and follow the directions there.
We use [version 0.1.23](https://github.com/dme65/pySOT/tree/306ece9785fcb90537c337e1555a71e9fbb6e6f1)

The implementation of the deep learning algorithms is in torch, so you would also need to install torch from [torch](https://github.com/torch/distro). 
We ran the deep learning algorithms on a cluster of GPU devices but you can modify the code and run them on CPU.



### Reproducing

After successfully installing pySOT and torch you can run the experiments by executing:

```
cd 6-MLP
python pySOT_runner.py 1 200 139 exp-1
```

6-MLP is the folder containing the code for experiment 6-MLP (as referenced in the paper). Other experiments are 8-CNN, 15-CNN, or 19-CNN.

The four arguments to `pySOT_runner.py` are the following:

- number of threads: we only tested the serial version of pySOT i.e. only one thread in all our experiments
- function evaluation budget: we limit the number of function evaluations (full training and validation of the deep learning algorithms) to 200 in all our experiments.
- seed number
- experiment name


### Citing the HORD algorithm

To cite the paper use the following BibTeX entry:

```
@inproceedings{ilievski2017efficient,
  title={Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates},
  author={Ilievski, Ilija and Akhtar, Taimoor and Feng, Jiashi and Shoemaker, Christine},
  booktitle={31st AAAI Conference on Artificial Intelligence (AAAI-17)},
  year={2017}
}
```


