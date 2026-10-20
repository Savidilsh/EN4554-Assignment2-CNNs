# EN4554 Assignment 2 — CNNs (PyTorch, CIFAR-10)

Assignment brief: [en4554_assignment_a2_cnn.pdf](en4554_assignment_a2_cnn.pdf)

| Task | Topic |
|------|-------|
| 1 | Backpropagation from scratch (Linear / ReLU / softmax-CE), gradient checks, MLP training, vanishing gradients |
| 2 | Optimizers (SGD+momentum, Adam) as `torch.optim.Optimizer` subclasses, 2-D trajectories, LR sensitivity |
| 3 | Regularization (weight decay, dropout, flip+crop, cutout, mixup) and plain vs residual nets (depth 8/20/56) |
| 4 | Best CNN ≤ 500k params, ≤ 20 epochs, ablations, single test evaluation |

Setup (all tasks): CIFAR-10 split 45k train / 5k val, per-channel normalization with
training statistics, fixed seeds; the test set is used once, in Task 4.
