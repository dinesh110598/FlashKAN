# FlashKAN: Grid size-independent computation of Kolmogorov Arnold networks using BSpline bases

Check out the [demo notebook](/demo.ipynb)

In short, we demonstrate how FlashKAN's training and inference speed scales much better with the grid size G compared to other BSpline-based implementations of the Kolmogorov Arnold Linear layer (in pytorch), without any shortcomings in the loss/accuracy performance with the example of training on the MNIST dataset. Memory consumption/allocations are yet to be benchmarked and could perhaps be better optimimized. More simply put, it's faster not just \<insert constant factor\> times but G times compared to existing implementations!