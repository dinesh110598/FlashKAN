# FlashKAN: Grid size-independent computation of Kolmogorov Arnold networks using BSpline bases

Check out the [demo notebook](/demo.ipynb)

In short, we demonstrate how FlashKAN's training and inference speed scales much better with the grid size G compared to other BSpline-based implementations of the Kolmogorov Arnold Linear layer (in pytorch), without any shortcomings in the loss/accuracy performance with the example of training on the MNIST dataset. Memory consumption/allocations are yet to be benchmarked and could perhaps be better optimimized.

While FlashKAN still has $\mathcal{O}(N^2L(G+k))$ (using notation from the [paper](https://arxiv.org/abs/2404.19756)) many parameters, we have reduced the training/inference speed to $\mathcal{O}(N^2Lk)$, making it $\mathcal{O}(G)$ faster than an MLP with similar number of parameters.

## References
- Liu, Ziming, et al. "Kan: Kolmogorov-arnold networks." arXiv preprint [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) (2024)
- [Efficient KAN](https://github.com/Blealtan/efficient-kan)
- BSplines basis function and derivative calculations heavily inspired from [BSplineKit.jl](https://github.com/jipolanco/BSplineKit.jl)