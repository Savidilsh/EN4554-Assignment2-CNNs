# EN4554 Assignment 2 — CNNs (PyTorch, CIFAR-10)

**Savindu Dilshan Wickramasinghe — 220701X**

Main deliverable: [a02_cnn.ipynb](a02_cnn.ipynb) (executed, with outputs). Source of the notebook: [a02_cnn.py](a02_cnn.py) (percent format, built with `tools/build_notebook.py`, executed with `tools/run_notebook.py`).

Assignment brief: [en4554_assignment_a2_cnn.pdf](en4554_assignment_a2_cnn.pdf)

| Task | Topic |
|------|-------|
| 1 | Backpropagation from scratch (Linear / ReLU / softmax-CE), gradient checks, MLP training, vanishing gradients |
| 2 | Optimizers (SGD+momentum, Adam) as `torch.optim.Optimizer` subclasses, 2-D trajectories, LR sensitivity |
| 3 | Regularization (weight decay, dropout, flip+crop, cutout, mixup) and plain vs residual nets (depth 8/20/56) |
| 4 | Best CNN ≤ 500k params, ≤ 20 epochs, ablations, single test evaluation |

Setup (all tasks): CIFAR-10 split 45k train / 5k val, per-channel normalization with
training statistics, fixed seeds; the test set is used once, in Task 4.

## Results (GTX 1650, fp32)

| Task | Result |
|------|--------|
| 1 | Manual backprop matches autograd (rel. err ~1e-16) and finite differences (~1e-9); manual MLP 52.0% val; sigmoid gradients vanish ~10^4x over 10 layers, ReLU+He stays flat |
| 2 | Own optimizers match `torch.optim` exactly; best val acc after 10 epochs: SGD 58.2% (lr 0.1), momentum 72.3% (lr 0.1), Adam 71.9% (lr 1e-3, fails at >=1e-2) |
| 3a | 5k images: baseline 68.2% val (32-pt gap), flip+crop 73.9%, all combined 67.0% (9-pt gap, underfits) |
| 3b | Plain-56 train loss 1.15 / 59.3% val vs ResNet-56 0.17 / 88.9% (degradation problem) |
| 4 | 460,548-param ResNet, 20 epochs: **90.36% test accuracy** (evaluated once) |

Report PDF: [220701X_a02.pdf](220701X_a02.pdf) (regenerate with `bash tools/make_pdf.sh`).
