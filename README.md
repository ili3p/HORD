# Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates

Code for reproducing results published in the paper "Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates" (AAAI-17) by Ilija Ilievski, Taimoor Akhtar, Jiashi Feng, and Christine Annette Shoemaker. 

### Installation

The HORD algorithm presented in the paper uses the open-source pySOT toolbox by David Eriksson, David Bindel, and Christine Shoemaker.
To install pySOT go to [pySOT](https://github.com/dme65/pySOT) and follow the directions there.

The implementation of the deep learning algorithms is in torch, so you would also need to install torch from [here](https://github.com/torch/distro). We ran the deep learning algorithms on a cluster of GPU devices but you can modify the code and run them on CPU.



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


### Citing the Hyperparameter Optimization via RBF and Dynamic coordinate search (HORD) algorithm

To cite the paper use the following BibTeX entry:

```
@inproceedings{melville2002content,
  title={Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates},
  author={Ilievski, Ilija and Akhtar, Taimoor and Feng, Jiashi and Shoemaker, Christine},
  booktitle={31st AAAI Conference on Artificial Intelligence (AAAI-17)},
  year={2017}
}
```


