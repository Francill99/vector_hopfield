# Vector Hopfield Model [![arXiv](https://img.shields.io/badge/arXiv-2507.02586-b31b1b.svg)](https://arxiv.org/abs/2507.02586)

Code to replicate results in paper: 

**"Statistical mechanics of vector Hopfield network near and above saturation"** 

Authors: *Flavio Nicoletti, Francesco D'Amico, Matteo Negri*

## Models

### VectorHopfield

$$H(\mathbf{s}) = - \sum_{i,j}J_{ij} \mathbf{s}_i \mathbf{s}_j$$

$$J_{ij} = \frac{1}{N} \sum_{\mu=1}^P \mathbf{x}_i^\mu \cdot \mathbf{x}_j^\mu$$

where $x_{\gamma i}^\mu$ are memories and 

$$ \mathbf{x} _i \cdot \mathbf{x} _j=\sum_{\gamma=1}^d x_{\gamma i}\,x_{\gamma j}$$

### TensorHopfield

$$H(\mathbf{s}) = - \sum_{\gamma,i,\delta,j}J_{\gamma i \delta j} s_{\gamma i} s_{\delta j}$$

$$J_{ij} = \frac{1}{N} \sum_{\mu=1}^P x_{\gamma i}^\mu x_{\delta j}^\mu$$

where $x_{\gamma i}^\mu$ are memories.

### Basic structure

* `src` contains the classes
* `run` contains scripts to run various experiments
* `plot` contains scripts to make plots quickly

### Usage

The file `hopfield.py` in `src` contains the class `VectorHopfield`, which implements the basic functions to create a dataset of random examples, build the Hebb rule and run the dynamics, both in the synchronous and asynchronous fashion.

The basic usage is to declare an instance of the class and call its method `run_dynamics()`:
```python
import src.hopfield as H

## generate the model
N=500; d=2; P=20
model = H.VectorHopfield(N,d,P)

## check if the first example is a stable fixed point
q, it = model.run_dynamics(model.examples[0,:,:])

## print the final magnetization q
print(N,d,P,q)
```

### Graphs presented in the paper are in two folders: 

* `graphs` contains notebooks and data to reproduce plots for Section 3 (below saturation)
* `notebooks_first_step` contains notebook and data to reproduce plots for Section 4 (above saturation)

Those notebooks contain both the code to produce the data with simulations and the data produced to reproduce the plots
