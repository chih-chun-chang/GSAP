# GSAP: GPU-Accelerated Stochastic Graph Partitioner

GSAP enhances the runtime performance of stochastic block partitioning(SBP) by using GPU. This project proposes a parallel algorithm to speed
up the generation process of stochastic proposals. and accelerate the calculation of the minimal description length. This project also introduces an efficient blockmodel update algorithm to dynamically manage the blockmodel matrix on GPU to achieve better performance. For more details, please refer to this [paper](https://tsung-wei-huang.github.io/papers/2024-ICPP-GSAP.pdf).

## Build
In this project, we use [moderngpu](https://github.com/moderngpu/moderngpu) and [bb_segsort](https://github.com/vtsynergy/bb_segsort/tree/master) for segmented operations.

```
git clone --recurse-submodules https://github.com/chih-chun-chang/GSAP
mkdir build && cd build
cmake ..
make
make test
```
## Dataset 
You can download datasets from the GraphChallenge website [here](http://graphchallenge.mit.edu/data-sets)

## Run
To run the project, download the dataset to the specified directory and then run 
```
./run [LL/LH/HL/HH] [1k/5k/20k/50k/200k]
```
