# GSAP
GPU-Accelerated Stochastic Graph Partitioner

## Build
In this project, we use [moderngpu](https://github.com/moderngpu/moderngpu) and [bb_segsort](https://github.com/vtsynergy/bb_segsort/tree/master) for the segmented operations.

```
git clone --recurse-submodules https://github.com/chih-chun-chang/GSAP
mkdir build && cd build
cmake ..
make
```

## Dataset 
You can download datasets from the GraphChallenge website [here](http://graphchallenge.mit.edu/data-sets)
