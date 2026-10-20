# %% [markdown]
# # EN4554 Assignment 2 — Convolutional Neural Networks
#
# **Name:** Savindu Dilshan Wickramasinghe &nbsp;&nbsp;&nbsp; **Index No.:** 220701X
#
# **GitHub repository:** https://github.com/Savidilsh/EN4554-Assignment2-CNNs
#
# **Acknowledgement of AI tool use.** Claude (Anthropic, via Claude Code) was used as a coding assistant
# to help write and debug the code in this notebook. I have read, run and understood all of the code and
# the interpretations below.
#
# All experiments use PyTorch and CIFAR-10. The 50k training images are split once (fixed seed) into
# 45k train / 5k validation, every image is normalized per channel with the mean/std of the 45k training
# images, and the test set is touched exactly once, at the very end of Task 4.

# %% [markdown]
# ## 0. Setup
#
# For speed (Colab T4 or a laptop GPU) the whole normalized dataset is kept on the GPU and mini-batches,
# random crop/flip, cutout and mixup are all done with tensor ops on the GPU, which avoids the CPU
# `DataLoader` bottleneck. Every run's history is cached as JSON in `results/` so that a crashed or
# interrupted session can be resumed without retraining finished runs.

# %%
import copy, json, math, os, random, time, warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")

SEED = 0
QUICK = os.environ.get("A2_QUICK", "0") == "1"  # smoke-test mode: a few batches per run
DATA_DIR = os.environ.get("A2_DATA", "./data")
RESULTS = "results_quick" if QUICK else "results"
FIGS = "figures_quick" if QUICK else "figures"
CKPT = "checkpoints"
for d in (RESULTS, FIGS, CKPT):
    os.makedirs(d, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = device.type == "cuda"  # fp16 autocast on GPU
torch.backends.cudnn.benchmark = True


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def savefig(name):
    plt.savefig(f"{FIGS}/{name}.png", dpi=130, bbox_inches="tight")


def cached(name, fn):
    """Run fn() once and cache its JSON-serialisable result in RESULTS/name.json."""
    path = f"{RESULTS}/{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = fn()
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def epochs(e):
    return 1 if QUICK else e


def log(msg):
    """Print and also append to RESULTS/progress.log (handy for following long runs)."""
    print(msg, flush=True)
    with open(f"{RESULTS}/progress.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


set_seed()
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
print("torch", torch.__version__, "| QUICK =", QUICK)

# %% [markdown]
# ### Data: 45k / 5k split and per-channel normalization with training statistics

# %%
train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
CLASSES = train_set.classes

X_all = torch.from_numpy(train_set.data).permute(0, 3, 1, 2).contiguous()  # uint8, N x 3 x 32 x 32
y_all = torch.tensor(train_set.targets)
perm = torch.randperm(len(y_all), generator=torch.Generator().manual_seed(SEED))
tr_idx, va_idx = perm[:45000], perm[45000:]

# per-channel statistics of the 45k training images only
MEAN = torch.stack([X_all[tr_idx, c].float().div(255).mean() for c in range(3)])
STD = torch.stack([X_all[tr_idx, c].float().div(255).std() for c in range(3)])


def normalize(X):
    return (X.float().div(255) - MEAN[:, None, None]) / STD[:, None, None]


X_tr, y_tr = normalize(X_all[tr_idx]).to(device), y_all[tr_idx].to(device)
X_va, y_va = normalize(X_all[va_idx]).to(device), y_all[va_idx].to(device)
# the test set stays on the CPU and is only used once, in Task 4
X_te = normalize(torch.from_numpy(test_set.data).permute(0, 3, 1, 2))
y_te = torch.tensor(test_set.targets)

print("train", tuple(X_tr.shape), "val", tuple(X_va.shape), "test", tuple(X_te.shape))
print("mean", MEAN.numpy().round(4), "std", STD.numpy().round(4))
print("val class counts", torch.bincount(y_va.cpu()).tolist())

# %% [markdown]
# ### Batching, GPU augmentation (flip + crop, cutout, mixup) and a generic training loop

# %%
def batches(X, y, bs, shuffle=True):
    n = len(y)
    idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for k, i in enumerate(range(0, n, bs)):
        if QUICK and k >= 3:
            break
        j = idx[i:i + bs]
        yield X[j], y[j]


def flip_crop(x, pad=4):
    """Random horizontal flip + random 32x32 crop from a zero-padded (= mean-padded) image."""
    N, C, H, W = x.shape
    flip = torch.rand(N, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    rows = (i[:, None] + torch.arange(H, device=x.device))[:, None, :, None]
    cols = (j[:, None] + torch.arange(W, device=x.device))[:, None, None, :]
    n_idx = torch.arange(N, device=x.device)[:, None, None, None]
    c_idx = torch.arange(C, device=x.device)[None, :, None, None]
    return xp[n_idx, c_idx, rows, cols]


def cutout(x, size=16):
    """Zero (= mean) a size x size square at a random centre of each image (may be clipped at the border)."""
    N, _, H, W = x.shape
    cy = torch.randint(0, H, (N, 1, 1), device=x.device)
    cx = torch.randint(0, W, (N, 1, 1), device=x.device)
    yy = torch.arange(H, device=x.device)[None, :, None]
    xx = torch.arange(W, device=x.device)[None, None, :]
    mask = (yy >= cy - size // 2) & (yy < cy + size // 2) & (xx >= cx - size // 2) & (xx < cx + size // 2)
    return x * (~mask)[:, None].to(x.dtype)


def mixup(x, y, alpha):
    """Return mixed inputs, both label sets and the mixing coefficient lambda ~ Beta(alpha, alpha)."""
    lam = float(np.random.beta(alpha, alpha))
    p = torch.randperm(len(y), device=x.device)
    return lam * x + (1 - lam) * x[p], y, y[p], lam


def compose(*fs):
    def f(x):
        for g in fs:
            x = g(x)
        return x
    return f


def cosine_warmup(opt, total_steps, warmup_steps=0):
    """Per-iteration LR multiplier: linear warm-up, then cosine decay to 0."""
    def mult(s):
        if s < warmup_steps:
            return (s + 1) / warmup_steps
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, mult)


@torch.no_grad()
def evaluate(model, X, y, bs=1000):
    model.eval()
    n = 1000 if QUICK else len(y)
    loss, correct = 0.0, 0
    for i in range(0, n, bs):
        xb, yb = X[i:i + bs].to(device), y[i:i + bs].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=AMP):
            out = model(xb)
        loss += F.cross_entropy(out.float(), yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss / n, correct / n


def fit(model, opt, n_epochs, X, y, bs=128, aug=None, mixup_alpha=0.0, sched=None,
        label_smoothing=0.0, amp=AMP, eval_train=False, tag=""):
    """Train for n_epochs; returns per-epoch train loss / val loss / val acc (and clean train acc)."""
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    H = dict(train_loss=[], val_loss=[], val_acc=[], train_acc=[], iter_loss=[], time=[])
    for ep in range(n_epochs):
        model.train()
        t0, losses = time.time(), []
        for xb, yb in batches(X, y, bs):
            if aug is not None:
                xb = aug(xb)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                if mixup_alpha > 0:
                    xb, ya, yb2, lam = mixup(xb, yb, mixup_alpha)
                    out = model(xb)
                    loss = lam * F.cross_entropy(out, ya) + (1 - lam) * F.cross_entropy(out, yb2)
                else:
                    out = model(xb)
                    loss = F.cross_entropy(out, yb, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            losses.append(loss.detach().float())
        il = torch.stack(losses).cpu().tolist()
        H["iter_loss"] += il
        H["train_loss"].append(float(np.mean(il)))
        vl, va = evaluate(model, X_va, y_va)
        H["val_loss"].append(vl); H["val_acc"].append(va)
        if eval_train:
            H["train_acc"].append(evaluate(model, X, y)[1])
        H["time"].append(time.time() - t0)
        msg = f"[{tag}] ep {ep + 1:3d}/{n_epochs} loss {H['train_loss'][-1]:.4f} val_acc {va:.4f}"
        if eval_train:
            msg += f" train_acc {H['train_acc'][-1]:.4f}"
        if ep == 0 or (ep + 1) % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1:
            log(msg + f" ({H['time'][-1]:.1f}s)")
        if not math.isfinite(H["train_loss"][-1]):
            log(f"[{tag}] diverged (non-finite loss), stopping early")
            break
    return H


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def smooth(v, k=50):
    v = np.asarray(v, dtype=float)
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")

# %% [markdown]
# ---
# ## Task 1: Backpropagation
#
# ### (a) Linear, ReLU and softmax cross-entropy with explicit forward/backward (no autograd)
#
# Each layer caches what it needs in `forward` and implements the chain rule in `backward`:
#
# * **Linear** $Y = XW + b$: $\;\partial L/\partial W = X^\top G$, $\;\partial L/\partial b = \sum_n G_n$, $\;\partial L/\partial X = G W^\top$, where $G = \partial L/\partial Y$.
# * **ReLU** $y=\max(0,x)$: $\;\partial L/\partial x = G \odot \mathbb 1[x>0]$.
# * **Sigmoid** (needed for 1(d)) $s=\sigma(x)$: $\;\partial L/\partial x = G \odot s(1-s)$.
# * **Softmax cross-entropy** with mean over the batch: $L = -\frac1N\sum_n \log p_{n,y_n}$, $\;\partial L/\partial z = (p - \text{onehot}(y))/N$.
#   The forward pass subtracts the row max (log-sum-exp trick) for numerical stability.

# %%
class Linear:
    def __init__(self, d_in, d_out, init="he", dtype=torch.float32, dev=device, gen=None):
        std = math.sqrt(2.0 / d_in) if init == "he" else math.sqrt(2.0 / (d_in + d_out))  # He or Xavier
        self.W = torch.randn(d_in, d_out, generator=gen, dtype=dtype).to(dev) * std
        self.b = torch.zeros(d_out, dtype=dtype, device=dev)
        self.dW = self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW = self.x.T @ g
        self.db = g.sum(0)
        return g @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask

    def params(self):
        return []


class Sigmoid:
    def forward(self, x):
        self.s = 1.0 / (1.0 + torch.exp(-x))
        return self.s

    def backward(self, g):
        return g * self.s * (1 - self.s)

    def params(self):
        return []


class SoftmaxCrossEntropy:
    def forward(self, z, y):
        z = z - z.max(dim=1, keepdim=True).values
        logp = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))
        self.p, self.y = torch.exp(logp), y
        return -logp[torch.arange(len(y)), y].mean()

    def backward(self):
        g = self.p.clone()
        g[torch.arange(len(self.y)), self.y] -= 1
        return g / len(self.y)


class MLP:
    def __init__(self, sizes, act="relu", init="he", dtype=torch.float32, dev=device, seed=SEED):
        gen = torch.Generator().manual_seed(seed)
        Act = ReLU if act == "relu" else Sigmoid
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(a, b, init, dtype, dev, gen))
            if i < len(sizes) - 2:
                self.layers.append(Act())
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        return self.loss_fn.forward(self.forward(x), y)

    def backward(self):
        g = self.loss_fn.backward()
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g  # dL/dx

    def linears(self):
        return [l for l in self.layers if isinstance(l, Linear)]

    def sgd_step(self, lr):
        for lin in self.linears():
            lin.W -= lr * lin.dW
            lin.b -= lr * lin.db

# %% [markdown]
# ### (b) Gradient check against autograd and centred finite differences
#
# A small MLP (12-10-8-5) in float64 on a random batch. For every parameter tensor we compare the manual
# gradient $g_m$ with (i) the autograd gradient of an equivalent `torch` graph sharing the same weights, and
# (ii) centred finite differences $\big(L(\theta+h e_i) - L(\theta-h e_i)\big)/2h$ with $h=10^{-6}$.
# The relative error is $\|g_m - g\|_2 / \max(\|g_m\|_2 + \|g\|_2, 10^{-12})$.
# The check is done for both the ReLU and the sigmoid network (the latter is used in 1(d)).

# %%
def rel_err(a, b):
    return ((a - b).norm() / max((a.norm() + b.norm()).item(), 1e-12)).item()


def grad_check(act):
    torch.manual_seed(1)
    net = MLP([12, 10, 8, 5], act=act, dtype=torch.float64, dev="cpu", seed=1)
    for lin in net.linears():  # non-zero biases so that their gradients are tested properly
        lin.b += 0.1 * torch.randn_like(lin.b)
    x = torch.randn(6, 12, dtype=torch.float64)
    y = torch.randint(0, 5, (6,))

    # manual gradients
    net.loss(x, y)
    dx_manual = net.backward()
    manual = {}
    for k, lin in enumerate(net.linears()):
        manual[f"W{k + 1}"], manual[f"b{k + 1}"] = lin.dW.clone(), lin.db.clone()
    manual["x"] = dx_manual

    # autograd on the same weights
    leaves = {}
    h = x.clone().requires_grad_(True)
    leaves["x"] = h
    lins = net.linears()
    for k, lin in enumerate(lins):
        W = lin.W.clone().requires_grad_(True); b = lin.b.clone().requires_grad_(True)
        leaves[f"W{k + 1}"], leaves[f"b{k + 1}"] = W, b
        h = h @ W + b
        if k < len(lins) - 1:
            h = torch.relu(h) if act == "relu" else torch.sigmoid(h)
    F.cross_entropy(h, y).backward()

    # centred finite differences
    def fd(tensor, eps=1e-6):
        g = torch.zeros_like(tensor)
        flat, gflat = tensor.view(-1), g.view(-1)
        for i in range(flat.numel()):
            old = flat[i].item()
            flat[i] = old + eps; lp = net.loss(x, y).item()
            flat[i] = old - eps; lm = net.loss(x, y).item()
            flat[i] = old
            gflat[i] = (lp - lm) / (2 * eps)
        return g

    rows = []
    for name, g_m in manual.items():
        if name == "x":
            target = x
        else:
            k = int(name[1:]) - 1
            target = lins[k].W if name[0] == "W" else lins[k].b
        g_auto = leaves[name].grad
        g_fd = fd(target)
        rows.append((act, name, tuple(g_m.shape), rel_err(g_m, g_auto), rel_err(g_m, g_fd),
                     torch.allclose(g_m, g_auto), torch.allclose(g_m, g_fd, atol=1e-7)))
    return rows


print(f"{'act':8s}{'param':6s}{'shape':>10s}{'rel.err vs autograd':>22s}{'rel.err vs FD':>16s}"
      f"{'allclose(auto)':>16s}{'allclose(FD)':>14s}")
for act in ("relu", "sigmoid"):
    for r in grad_check(act):
        print(f"{r[0]:8s}{r[1]:6s}{str(r[2]):>10s}{r[3]:22.2e}{r[4]:16.2e}{str(r[5]):>16s}{str(r[6]):>14s}")

# %% [markdown]
# **Interpretation.** For every parameter tensor (and for the input gradient) the manual backward pass
# agrees with autograd to a relative error of about $10^{-16}$, i.e. machine precision in float64: the two
# compute the same expression, only the order of floating-point operations differs. Against centred finite
# differences the relative error is $10^{-10}$–$10^{-8}$, which is exactly what we expect from the method:
# the truncation error is $O(h^2)\approx10^{-12}$ and the round-off error is $O(\epsilon_{mach}/h)\approx10^{-10}$.
# `torch.allclose` returns `True` in all cases, for both the ReLU and the sigmoid network, so the hand-written
# Linear / ReLU / Sigmoid / softmax cross-entropy backward passes are correct. (ReLU is not differentiable at 0,
# but with random real-valued inputs no pre-activation lands within $h$ of 0, so finite differences are valid.)

# %% [markdown]
# ### (c) Training the MLP with SGD on flattened CIFAR-10
#
# MLP 3072 → 512 → 256 → 10 with ReLU, He initialization, mini-batch SGD (batch 128, learning rate 0.05
# with cosine decay), 30 epochs, using only the hand-written forward/backward above (no autograd).

# %%
def train_manual_mlp(n_epochs=30, lr0=0.05, bs=128):
    set_seed()
    net = MLP([3072, 512, 256, 10], act="relu", init="he")
    Xf_tr, Xf_va = X_tr.reshape(len(X_tr), -1), X_va.reshape(len(X_va), -1)
    steps = n_epochs * math.ceil(len(y_tr) / bs)
    H = dict(iter_loss=[], train_loss=[], val_loss=[], val_acc=[], train_acc=[])
    step = 0
    for ep in range(n_epochs):
        losses = []
        for xb, yb in batches(Xf_tr, y_tr, bs):
            lr = lr0 * 0.5 * (1 + math.cos(math.pi * step / steps))
            losses.append(net.loss(xb, yb))
            net.backward()
            net.sgd_step(lr)
            step += 1
        il = torch.stack(losses).cpu().tolist()
        H["iter_loss"] += il
        H["train_loss"].append(float(np.mean(il)))

        def acc_loss(Xf, y):
            n = 1000 if QUICK else len(y)
            out = torch.cat([net.forward(Xf[i:i + 1000]) for i in range(0, n, 1000)])
            return net.loss_fn.forward(out, y[:n]).item(), (out.argmax(1) == y[:n]).float().mean().item()

        vl, va = acc_loss(Xf_va, y_va)
        H["val_loss"].append(vl); H["val_acc"].append(va)
        H["train_acc"].append(acc_loss(Xf_tr, y_tr)[1])
        log(f"ep {ep + 1:2d} train_loss {H['train_loss'][-1]:.4f} train_acc {H['train_acc'][-1]:.4f} "
              f"val_loss {vl:.4f} val_acc {va:.4f}")
    return H


H1c = cached("task1c_mlp", lambda: train_manual_mlp(epochs(30)))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(H1c["iter_loss"], alpha=0.25, lw=0.5, label="per iteration")
ax[0].plot(np.arange(len(smooth(H1c["iter_loss"]))) + 25, smooth(H1c["iter_loss"]), label="moving avg (50)")
ax[0].set(xlabel="iteration", ylabel="cross-entropy", title="Training loss (per iteration)"); ax[0].legend()
ep = np.arange(1, len(H1c["train_loss"]) + 1)
ax[1].plot(ep, H1c["train_loss"], "o-", label="train"); ax[1].plot(ep, H1c["val_loss"], "s-", label="validation")
ax[1].set(xlabel="epoch", ylabel="cross-entropy", title="Loss per epoch"); ax[1].legend()
ax[2].plot(ep, H1c["train_acc"], "o-", label="train"); ax[2].plot(ep, H1c["val_acc"], "s-", label="validation")
ax[2].set(xlabel="epoch", ylabel="accuracy", title="Accuracy"); ax[2].legend()
plt.tight_layout(); savefig("task1c_mlp"); plt.show()
print(f"Final validation accuracy: {H1c['val_acc'][-1]:.4f} (best {max(H1c['val_acc']):.4f})")

# %% [markdown]
# **Interpretation.** The hand-written MLP trains correctly with SGD: the training loss falls smoothly from
# 2.3 to about 0.11 (the small steps in the per-iteration curve are epoch boundaries, where the reshuffled data
# starts again) and the final **validation accuracy is 52.0%** (best 52.0%), far above the 10% chance level and
# typical for an MLP on raw CIFAR-10 pixels. However, the model overfits heavily: train accuracy reaches 98.9%,
# while the validation loss is lowest at epoch 5 (1.48) and then *increases* to 1.97 although the validation accuracy
# stays flat around 51–52%: the network becomes over-confident on the validation images it gets wrong. An MLP on
# flattened pixels has no notion of locality or translation invariance (1.7M parameters, each pixel treated as an
# independent feature), so it memorises the training set instead of learning features that generalise; this is the
# motivation for the convolutional networks in the following tasks (which reach >70% after only a few epochs).

# %% [markdown]
# ### (d) Per-layer gradient norms at initialization in a 10-layer MLP: sigmoid vs ReLU
#
# A 10-layer MLP (3072 → 9 × 256 hidden → 10) is evaluated at initialization on a batch of 512 real
# training images, and $\|\partial L/\partial W_\ell\|_F$ is recorded for every layer $\ell$ (layer 1 is next to
# the input). We use the same network with the activation as the only change, for two standard
# initializations (He: $\mathrm{Var}(W)=2/d_{in}$; Xavier: $\mathrm{Var}(W)=2/(d_{in}+d_{out})$), averaged
# over 5 random initializations.

# %%
def layer_grad_norms(act, init, seeds=range(5)):
    xb, yb = X_tr[:512].reshape(512, -1), y_tr[:512]
    norms = []
    for s in seeds:
        net = MLP([3072] + [256] * 9 + [10], act=act, init=init, seed=s)
        net.loss(xb, yb)
        net.backward()
        norms.append([lin.dW.norm().item() for lin in net.linears()])
    return np.mean(norms, axis=0)


fig, ax = plt.subplots(figsize=(7.5, 4.5))
for act, c in (("sigmoid", "tab:red"), ("relu", "tab:blue")):
    for init, ls in (("he", "-"), ("xavier", "--")):
        g = layer_grad_norms(act, init)
        ax.semilogy(range(1, 11), g, ls, color=c, marker="o", label=f"{act}, {init} init")
        print(f"{act:8s}{init:7s}", " ".join(f"{v:.1e}" for v in g))
ax.set(xlabel="layer (1 = input side, 10 = output side)", ylabel=r"$\|\partial L/\partial W_\ell\|_F$ (log)",
       title="Per-layer gradient norm at initialization, 10-layer MLP")
ax.set_xticks(range(1, 11)); ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.tight_layout(); savefig("task1d_gradnorms"); plt.show()

# %% [markdown]
# **Explanation.** Backpropagation multiplies the incoming gradient, layer by layer, by $W_\ell^\top$ and by the
# derivative of the activation: $\delta_{\ell} = (W_{\ell+1}^\top \delta_{\ell+1}) \odot \phi'(z_\ell)$, and
# $\partial L/\partial W_\ell = h_{\ell-1}^\top\delta_\ell$.
#
# * **Sigmoid:** $\sigma'(z) = \sigma(z)(1-\sigma(z)) \le 0.25$, so every layer shrinks the gradient by a factor
#   of roughly 3–5 (the weights cannot compensate). Over 9 activations the gradient norm decays geometrically,
#   which is a straight line on the log plot: the first layer receives a gradient about $10^4$ (He init) to
#   $10^5$ (Xavier) times smaller than the last layer. The early layers would therefore barely learn with SGD:
#   this is the **vanishing-gradient problem**. Sigmoid outputs are also not zero-centred, which makes it worse.
# * **ReLU:** $\phi'(z)\in\{0,1\}$ and, for symmetric pre-activations, half of the units are active, which halves
#   the variance of the back-propagated signal. He initialization ($\mathrm{Var}(W)=2/d_{in}$) compensates exactly for
#   this factor $1/2$, so the gradient norm stays roughly constant across all ten layers (the flat solid blue line).
#   With Xavier initialization and ReLU the hidden weights have variance $1/d_{in}$ (half of He's), so both the
#   forward activations $h_\ell$ (which shrink with depth) and the back-propagated $\delta_\ell$ (which shrink towards
#   the input) lose a factor $\sqrt2$ per layer. In $\partial L/\partial W_\ell = h_{\ell-1}^\top\delta_\ell$ the two effects
#   cancel, so the weight-gradient norm is flat but about 100× smaller than with He init, i.e. the whole network
#   learns slowly and the signal itself (activations and $\delta$) still vanishes exponentially with depth.
#
# The first-layer value for ReLU is slightly higher simply because $W_1$ has $3072\times256$ entries (the Frobenius
# norm grows with the number of entries). Conclusion: ReLU together with variance-preserving (He) initialization keeps
# gradients well-scaled at initialization, which is why deep networks use it (and batch norm / residual connections,
# Task 3, to keep it that way during training).

# %% [markdown]
# ---
# ## Task 2: Optimizers
#
# ### (a) SGD with momentum and Adam (with bias correction) as `torch.optim.Optimizer` subclasses
#
# * **SGD + momentum** (PyTorch convention, no dampening): $g \leftarrow \nabla_\theta L + \lambda\theta$,
#   $\;b_t = \mu b_{t-1} + g$ (with $b_1 = g$), $\;\theta \leftarrow \theta - \eta\, b_t$.
# * **Adam**: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g$, $\;v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2$,
#   bias correction $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$,
#   $\;\theta \leftarrow \theta - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$.

# %%
class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad if wd == 0 else p.grad + wd * p
                if mu != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()
                    else:
                        state["momentum_buffer"].mul_(mu).add_(g)
                    g = state["momentum_buffer"]
                p.add_(g, alpha=-lr)
        return loss


class MyAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad if wd == 0 else p.grad + wd * p
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss

# %% [markdown]
# **Equivalence test.** Two copies of the same small CNN (float64) are trained for 50 steps on the same
# CIFAR-10 mini-batches, one with our optimizer and one with the PyTorch reference; all parameters are
# compared at the end with `torch.allclose` (and the max absolute difference is reported, next to how far
# the parameters moved, to show that the comparison is not trivial).

# %%
def compare_optimizers(make_mine, make_ref, steps=50):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(), nn.Linear(16, 10)).double().to(device)
    a, b = copy.deepcopy(net), copy.deepcopy(net)
    oa, ob = make_mine(a.parameters()), make_ref(b.parameters())
    for s in range(steps):
        xb = X_tr[s * 64:(s + 1) * 64].double(); yb = y_tr[s * 64:(s + 1) * 64]
        for model, opt in ((a, oa), (b, ob)):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    diff = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    same = all(torch.allclose(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    moved = max((pa - p0).abs().max().item() for pa, p0 in zip(a.parameters(), net.parameters()))
    return same, diff, moved


tests = {
    "SGD (no momentum)": (lambda p: MySGD(p, lr=0.05), lambda p: torch.optim.SGD(p, lr=0.05)),
    "SGD momentum 0.9": (lambda p: MySGD(p, lr=0.05, momentum=0.9),
                         lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    "SGD momentum 0.9 + wd 5e-4": (lambda p: MySGD(p, lr=0.05, momentum=0.9, weight_decay=5e-4),
                                   lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=5e-4)),
    "Adam": (lambda p: MyAdam(p, lr=1e-3), lambda p: torch.optim.Adam(p, lr=1e-3)),
    "Adam betas (0.8, 0.99) + wd 1e-3": (lambda p: MyAdam(p, lr=3e-3, betas=(0.8, 0.99), weight_decay=1e-3),
                                         lambda p: torch.optim.Adam(p, lr=3e-3, betas=(0.8, 0.99),
                                                                    weight_decay=1e-3)),
}
print(f"{'optimizer':36s}{'allclose':>10s}{'max |diff|':>14s}{'max |change|':>14s}")
for name, (mine, ref) in tests.items():
    same, diff, moved = compare_optimizers(mine, ref)
    print(f"{name:36s}{str(same):>10s}{diff:14.2e}{moved:14.2e}")

# %% [markdown]
# **Interpretation.** After 50 steps on real CIFAR-10 batches, the parameters produced by `MySGD` and `MyAdam`
# are identical to those of `torch.optim.SGD` / `torch.optim.Adam` (`allclose = True`), with a maximum absolute
# difference of 0 for SGD and $\sim10^{-16}$ (float64 round-off) for the variants with weight decay and for Adam,
# while the parameters themselves moved by $5\times10^{-2}$–$3\times10^{-1}$, so the test is not trivial. The tiny
# Adam differences come only from the order of operations (PyTorch folds the bias corrections into the step size
# and the denominator, $\theta \leftarrow \theta - \frac{\eta}{1-\beta_1^t}\, m_t / (\sqrt{v_t}/\sqrt{1-\beta_2^t} + \epsilon)$,
# which is algebraically the same as our $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$). The momentum buffer convention
# ($b_1 = g_1$, no dampening) and the L2-style weight decay added to the gradient also match PyTorch.

# %% [markdown]
# ### (b) Trajectories on $f(x, y) = \tfrac12 (x^2 + 50 y^2)$
#
# $\nabla f = (x,\; 50y)$, so the Hessian is $\mathrm{diag}(1, 50)$ (condition number 50). Plain gradient
# descent is stable only for $\eta < 2/50 = 0.04$. All three methods start at $(-9, 1.5)$, run for 100 steps
# and use our own optimizer classes from (a); the gradient is computed by autograd.

# %%
def f_quad(p):
    return 0.5 * (p[0] ** 2 + 50 * p[1] ** 2)


def trajectory(make_opt, start=(-9.0, 1.5), steps=100):
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = make_opt([p])
    path = [p.detach().clone().numpy()]
    for _ in range(steps):
        opt.zero_grad()
        f_quad(p).backward()
        opt.step()
        path.append(p.detach().clone().numpy())
    return np.array(path)


runs_2b = {
    r"SGD, $\eta=0.035$": (lambda p: MySGD(p, lr=0.035), "tab:red"),
    r"SGD + momentum, $\eta=0.01, \mu=0.9$": (lambda p: MySGD(p, lr=0.01, momentum=0.9), "tab:blue"),
    r"Adam, $\eta=0.3$": (lambda p: MyAdam(p, lr=0.3), "tab:green"),
}
paths = {k: trajectory(v[0]) for k, v in runs_2b.items()}

xs, ys = np.meshgrid(np.linspace(-10, 10, 400), np.linspace(-3, 3, 300))
fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw=dict(width_ratios=[2, 1]))
ax[0].contour(xs, ys, 0.5 * (xs ** 2 + 50 * ys ** 2), levels=np.logspace(-1, 2.6, 18), cmap="Greys",
              linewidths=0.8)
for k, path in paths.items():
    ax[0].plot(path[:, 0], path[:, 1], "o-", ms=2.5, lw=1, color=runs_2b[k][1], label=k)
    ax[1].semilogy([f_quad(q) for q in path], color=runs_2b[k][1], label=k)
ax[0].plot(0, 0, "k*", ms=12)
ax[0].set(xlabel="x", ylabel="y", title=r"Trajectories on $f(x,y)=\frac{1}{2}(x^2+50y^2)$ (100 steps)",
          xlim=(-10, 10), ylim=(-3, 3))
ax[0].legend(loc="upper right")
ax[1].set(xlabel="iteration", ylabel="f (log)", title="Objective vs iteration"); ax[1].grid(alpha=0.3)
plt.tight_layout(); savefig("task2b_trajectories"); plt.show()
for k, path in paths.items():
    print(f"{k:42s} final point ({path[-1, 0]: .4f}, {path[-1, 1]: .4f})  f = {f_quad(path[-1]):.2e}")

# %% [markdown]
# **Explanation.** Gradient descent on this quadratic acts on each coordinate separately:
# $x_{t+1} = (1-\eta)x_t$ and $y_{t+1} = (1-50\eta)y_t$.
#
# * **SGD** ($\eta=0.035$, close to the stability limit $2/50$): along the steep $y$ direction the factor is
#   $1-50\eta=-0.75$, so $y$ flips sign every step (zig-zag across the valley) while it decays; along the shallow
#   $x$ direction the factor is $0.965$, so progress is very slow. The learning rate is limited by the *largest*
#   curvature, while the convergence speed is set by the *smallest* one: after 100 steps it is still at $x=-0.26$
#   ($f=3\times10^{-2}$). This is the classic ill-conditioning problem (condition number 50).
# * **SGD + momentum** ($\eta=0.01$, $\mu=0.9$): the velocity accumulates the consistent $x$-component of the
#   gradient, so the effective step along the valley is $\eta/(1-\mu)=0.1$, ten times larger, while the
#   alternating $y$-components largely cancel. The trajectory is under-damped (it overshoots the minimum and
#   oscillates in both coordinates, the "heavy ball" behaviour), but the per-step contraction is $\sqrt{\mu}\approx0.95$
#   in *both* directions instead of $0.965$ in $x$, so it reaches the lowest objective ($f\approx7\times10^{-4}$).
# * **Adam** ($\eta=0.3$): the update $\hat m/\sqrt{\hat v}$ divides each coordinate by its own gradient magnitude,
#   so the first steps have size $\approx\eta$ in both $x$ and $y$ (the path starts along the diagonal regardless
#   of the 50× different curvatures). This per-parameter scaling removes most of the ill-conditioning: $y$ is
#   driven to 0 quickly and $x$ moves at a roughly constant speed, giving a smooth curved path. Near the minimum
#   the normalised step does not shrink with the gradient, so Adam oscillates in a small region ($f\approx3\times10^{-3}$)
#   unless the learning rate is decayed.

# %% [markdown]
# ### (c) A small CNN trained with each optimizer at three learning rates
#
# Small CNN without batch norm (so that the optimizer, not normalization, determines stability):
# `[conv3-32, conv3-32, pool] → [conv3-64, conv3-64, pool] → [conv3-128, pool] → GAP → FC-10`, ReLU, about
# 0.14M parameters. Each of the 9 runs uses the same initialization (seed), the same data order, batch size
# 128, a constant learning rate from the common grid $\{10^{-3}, 10^{-2}, 10^{-1}\}$, 10 epochs, no
# augmentation, full fp32 precision, and our optimizer implementations from (a) (momentum 0.9; Adam with
# default betas).

# %%
class SmallCNN(nn.Module):
    def __init__(self, bn=False, p_drop=0.0, widths=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = widths

        def conv(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            return layers + [nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *conv(3, c1), *conv(c1, c1), nn.MaxPool2d(2), nn.Dropout(p_drop),
            *conv(c1, c2), *conv(c2, c2), nn.MaxPool2d(2), nn.Dropout(p_drop),
            *conv(c2, c3), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop), nn.Linear(c3, 10))

    def forward(self, x):
        return self.head(self.features(x))


print("SmallCNN parameters:", n_params(SmallCNN()))

OPTS_2C = {
    "SGD": lambda p, lr: MySGD(p, lr=lr),
    "SGD+momentum": lambda p, lr: MySGD(p, lr=lr, momentum=0.9),
    "Adam": lambda p, lr: MyAdam(p, lr=lr),
}
LRS_2C = [1e-3, 1e-2, 1e-1]


def run_2c(opt_name, lr):
    set_seed()
    model = SmallCNN().to(device)
    opt = OPTS_2C[opt_name](model.parameters(), lr)
    return fit(model, opt, epochs(10), X_tr, y_tr, bs=128, amp=False, tag=f"{opt_name} lr={lr:g}")


R2c = {(o, lr): cached(f"task2c_{o}_lr{lr:g}", lambda o=o, lr=lr: run_2c(o, lr))
       for o in OPTS_2C for lr in LRS_2C}

# %%
def final_acc(H):
    return H["val_acc"][-1] if math.isfinite(H["train_loss"][-1]) else 0.1


fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
colors = dict(zip(OPTS_2C, ["tab:red", "tab:blue", "tab:green"]))
print(f"{'optimizer':14s}" + "".join(f"{'lr=' + format(lr, 'g'):>12s}" for lr in LRS_2C) + "   (final val acc)")
for o in OPTS_2C:
    accs = [final_acc(R2c[(o, lr)]) for lr in LRS_2C]
    print(f"{o:14s}" + "".join(f"{a:12.4f}" for a in accs))
    ax[0].semilogx(LRS_2C, accs, "o-", color=colors[o], label=o)
    best = LRS_2C[int(np.argmax(accs))]
    ax[1].plot(smooth(R2c[(o, best)]["iter_loss"], 50), color=colors[o], label=f"{o} (best lr={best:g})")
ax[0].set(xlabel="learning rate (log)", ylabel="validation accuracy after 10 epochs",
          title="Validation accuracy vs learning rate"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].set(xlabel="iteration", ylabel="training loss (moving avg, 50 it)",
          title="Training loss at the best learning rate")
ax[1].grid(alpha=0.3); ax[1].legend()
plt.tight_layout(); savefig("task2c_lr_sensitivity"); plt.show()

# %% [markdown]
# **Discussion of sensitivity.** Final validation accuracy after 10 epochs:
#
# | optimizer | lr = 0.001 | lr = 0.01 | lr = 0.1 |
# |---|---|---|---|
# | SGD | 10.6% | 34.4% | **58.2%** |
# | SGD + momentum 0.9 | 35.1% | 68.5% | **72.3%** |
# | Adam | **71.9%** | 10.1% (stuck) | 10.1% (diverged, loss up to $2\times10^5$) |
#
# * **Plain SGD** needs a large learning rate: at $10^{-3}$ it does not leave the initial plateau (loss stays at
#   2.30) within 10 epochs, and accuracy increases monotonically with the learning rate over the grid, i.e. the
#   best value may even lie above 0.1. It is the slowest optimizer (the training loss at its best rate is still 1.10).
# * **Momentum** behaves like SGD with a 10× larger effective learning rate $\eta/(1-\mu)$: momentum at $\eta$
#   matches plain SGD at $10\eta$ (35.1% vs 34.4%; 68.5% vs 58.2%). It is good over a wider range (68–72% for
#   $\eta\in[0.01,0.1]$) and gives the lowest training loss and best accuracy at $\eta=0.1$.
# * **Adam** is the fastest at the start (the steepest loss decrease in the right plot, and the best first epoch),
#   but it is the *most sensitive* optimizer: it only works at $10^{-3}$. Because Adam's step size is about $\eta$ per
#   parameter regardless of the gradient scale, $\eta=10^{-2}$ already moves every weight by 0.01 per step, which
#   destroys the initial features of this network without batch norm (the ReLUs die and the loss sticks at
#   $\ln 10 = 2.30$), and $\eta=0.1$ makes the loss explode.
#
# So the "best" learning rate differs by two orders of magnitude between optimizers (0.1 for SGD/momentum, 0.001
# for Adam) and must be tuned per optimizer; comparing optimizers at a single shared learning rate would be
# misleading. With its best rate, SGD with momentum slightly outperforms Adam in final accuracy here, while Adam
# is the easiest to get working quickly with its default $10^{-3}$.

# %% [markdown]
# ---
# ## Task 3: Regularization and residual connections
#
# ### (a) Regularization on a 5k-image training subset
#
# Model: the same small CNN as in 2(c) with batch norm after every convolution (dropout layers are present
# but only active when $p>0$). Training data: the first 5,000 images of the 45k training split (classes
# roughly balanced); validation: the usual 5k validation split. All runs use the same seed, SGD with
# momentum 0.9, peak learning rate 0.05 with cosine decay, batch size 128 and 100 epochs. Train accuracy is
# measured on the *clean* (non-augmented) 5k training images in eval mode after every epoch, so it is
# comparable across methods.
#
# | variant | setting |
# |---|---|
# | baseline | no regularization |
# | weight decay | L2 penalty $5\times10^{-4}$ (coupled, via the optimizer) |
# | dropout | $p = 0.3$ after each pooling stage and before the classifier |
# | flip + crop | random horizontal flip + random 32×32 crop of the 4-pixel padded image |
# | cutout | one random 16×16 square zeroed per image (own implementation, `cutout` above) |
# | mixup | $\tilde x = \lambda x_i + (1-\lambda)x_j$, loss $\lambda\,\mathrm{CE}(y_i) + (1-\lambda)\,\mathrm{CE}(y_j)$, $\lambda\sim\mathrm{Beta}(0.4, 0.4)$ (own implementation, `mixup` above) |
# | all combined | all of the above together |

# %%
X_small, y_small = X_tr[:5000], y_tr[:5000]
print("5k subset class counts:", torch.bincount(y_small.cpu()).tolist())

VARIANTS_3A = {
    "baseline": dict(),
    "weight decay": dict(wd=5e-4),
    "dropout": dict(p_drop=0.3),
    "flip+crop": dict(aug=flip_crop),
    "cutout": dict(aug=cutout),
    "mixup": dict(mixup_alpha=0.4),
    "all combined": dict(wd=5e-4, p_drop=0.3, aug=compose(flip_crop, cutout), mixup_alpha=0.4),
}


def run_3a(name, n_epochs=100):
    cfg = VARIANTS_3A[name]
    set_seed()
    model = SmallCNN(bn=True, p_drop=cfg.get("p_drop", 0.0)).to(device)
    opt = MySGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=cfg.get("wd", 0.0))
    steps = n_epochs * math.ceil(len(y_small) / 128)
    return fit(model, opt, n_epochs, X_small, y_small, bs=128, aug=cfg.get("aug"),
               mixup_alpha=cfg.get("mixup_alpha", 0.0), sched=cosine_warmup(opt, steps), eval_train=True,
               tag=name)


R3a = {k: cached(f"task3a_{k.replace(' ', '_').replace('+', '_')}", lambda k=k: run_3a(k, epochs(100)))
       for k in VARIANTS_3A}

# %%
print(f"{'variant':14s}{'train acc':>11s}{'val acc':>10s}{'gap':>8s}{'best val':>10s}")
for k, H in R3a.items():
    tr, va = H["train_acc"][-1], H["val_acc"][-1]
    print(f"{k:14s}{tr:11.4f}{va:10.4f}{tr - va:8.4f}{max(H['val_acc']):10.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
cmap = plt.get_cmap("tab10")
for i, (k, H) in enumerate(R3a.items()):
    ep = np.arange(1, len(H["val_acc"]) + 1)
    axes[0].plot(ep, H["train_acc"], color=cmap(i), label=k)
    axes[1].plot(ep, H["val_acc"], color=cmap(i), label=k)
    axes[2].plot(ep, H["val_loss"], color=cmap(i), label=k)
axes[0].set(xlabel="epoch", ylabel="accuracy", title="Train accuracy (clean 5k train images)")
axes[1].set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy")
axes[2].set(xlabel="epoch", ylabel="cross-entropy", title="Validation loss", ylim=(0.5, 3.5))
for a in axes:
    a.grid(alpha=0.3)
axes[1].legend(loc="lower right")
plt.tight_layout(); savefig("task3a_regularization"); plt.show()

fig, ax = plt.subplots(figsize=(9, 3.8))
names = list(R3a)
tr = [R3a[k]["train_acc"][-1] for k in names]; va = [R3a[k]["val_acc"][-1] for k in names]
xpos = np.arange(len(names))
ax.bar(xpos - 0.2, tr, 0.4, label="train"); ax.bar(xpos + 0.2, va, 0.4, label="validation")
for x_, t_, v_ in zip(xpos, tr, va):
    ax.text(x_, max(t_, v_) + 0.01, f"gap {t_ - v_:.2f}", ha="center", fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(names); ax.set_ylim(0, 1.1); ax.set_ylabel("final accuracy")
ax.set_title("Final train vs validation accuracy (5k training images)"); ax.legend(loc="lower left")
plt.tight_layout(); savefig("task3a_gap"); plt.show()

# %% [markdown]
# **Interpretation.** Results after 100 epochs on 5k images (train accuracy on clean training images):
#
# | variant | train acc | val acc | gap | final val loss |
# |---|---|---|---|---|
# | baseline | 100.0% | 68.2% | 31.8 | 1.41 |
# | weight decay | 100.0% | 67.9% | 32.1 | 1.26 |
# | dropout | 98.2% | 70.1% | 28.1 | 0.97 |
# | flip + crop | 96.8% | **73.9%** | 22.9 | 0.95 |
# | cutout | 100.0% | 71.1% | 28.9 | 1.20 |
# | mixup | 100.0% | 68.6% | 31.4 | 0.97 |
# | all combined | 76.2% | 67.0% | **9.2** | **0.93** |
#
# * With only 5k images the **baseline** memorises the training set within ~35 epochs (100% train accuracy) and
#   stalls at 68% validation accuracy: a 32-point generalisation gap. Its validation loss is also the highest and
#   very noisy, because the network becomes over-confident.
# * **Flip + crop** is the most effective single regulariser (+5.7 points, smallest gap among single methods):
#   it encodes real invariances of natural images (left–right symmetry, small translations), effectively
#   multiplying the size of the training set. **Cutout** (+2.9) forces the network to use several parts of the
#   object instead of a single discriminative patch, but on its own the network still reaches 100% on clean
#   training images. **Dropout** (+1.9) reduces co-adaptation of features and is the only single method besides
#   flip+crop that keeps train accuracy below 100%.
# * **Weight decay** barely changes accuracy here. With batch norm after every conv the network output is invariant
#   to the scale of the conv weights, so L2 mainly increases the effective learning rate rather than restricting
#   capacity; it does lower the validation loss (1.41 → 1.26). **Mixup** gives a similar accuracy to the baseline, but
#   the validation loss drops from 1.41 to 0.97: training on interpolated images with soft targets makes the predictions
#   much better calibrated (less over-confident), even though the arg-max decisions change little.
# * **All combined** has by far the smallest gap (9 points) and the lowest validation loss, but it **underfits**: the
#   train accuracy only reaches 76% and validation accuracy (67%) is lower than with flip+crop alone. Stacking five
#   regularisers on a 0.14M-parameter network trained for only 4k iterations removes too much effective capacity;
#   the curves are still rising at epoch 100, so a bigger model or a longer schedule would be needed to benefit from it.
#
# In short, regularisation trades training fit for generalisation; the best choice depends on the capacity/data/compute
# balance, and data augmentation that reflects true invariances (flip + crop) is the most valuable when data is scarce.

# %% [markdown]
# ### (b) Plain vs residual CIFAR networks of depth $6n+2$ (8, 20, 56)
#
# Architecture of He et al. (2016) for CIFAR: a 3×3 conv with 16 channels, then three stages of $n$ basic
# blocks (two 3×3 conv–BN layers each) with 16/32/64 channels at resolutions 32/16/8 (the first conv of
# stages 2 and 3 has stride 2), global average pooling and a 10-way FC layer; $6n+2$ weighted layers in total.
# Kaiming-normal initialization (fan-out, ReLU) for all convolutions, BN with $\gamma=1, \beta=0$.
#
# The **only** difference between the two families is the shortcut: residual blocks compute
# $\mathrm{ReLU}(F(x) + s(x))$ and plain blocks $\mathrm{ReLU}(F(x))$. We use the parameter-free option-A
# shortcut (identity; when the shape changes, stride-2 subsampling and zero-padding of the extra channels),
# so plain and residual networks of the same depth have *exactly* the same parameters.
#
# Training (identical for all 6 networks): SGD, momentum 0.9, weight decay $10^{-4}$, batch 128, peak learning
# rate 0.1 with 1 epoch of linear warm-up and cosine decay, flip + crop augmentation, 20 epochs on the 45k split
# (mixed precision on the GPU).

# %%
class ShortcutA(nn.Module):
    """Identity shortcut; on a shape change, subsample spatially and zero-pad the new channels."""
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.stride, self.pad = stride, c_out - c_in

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return x


class ShortcutB(nn.Sequential):
    """Projection shortcut (1x1 conv + BN) on a shape change, identity otherwise."""
    def __init__(self, c_in, c_out, stride):
        if stride == 1 and c_in == c_out:
            super().__init__()
        else:
            super().__init__(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride, residual=True, shortcut="A"):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.residual = residual
        if residual:
            self.shortcut = (ShortcutA if shortcut == "A" else ShortcutB)(c_in, c_out, stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class CifarNet(nn.Module):
    """CIFAR ResNet / plain net: stem + 3 stages + GAP + FC. blocks=(n, n, n) gives depth 6n+2."""
    def __init__(self, blocks=(3, 3, 3), widths=(16, 32, 64), residual=True, shortcut="A", zero_init_last_bn=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        layers, c_in = [], widths[0]
        for s, (n, w) in enumerate(zip(blocks, widths)):
            for b in range(n):
                layers.append(BasicBlock(c_in, w, 2 if (s > 0 and b == 0) else 1, residual, shortcut))
                c_in = w
        self.stages = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
        if zero_init_last_bn and residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def depth_net(depth, residual):
    n = (depth - 2) // 6
    return CifarNet(blocks=(n, n, n), residual=residual)


for d in (8, 20, 56):
    print(f"depth {d:2d}: params plain {n_params(depth_net(d, False)):,}  residual {n_params(depth_net(d, True)):,}")

# %%
def run_3b(depth, residual, n_epochs=20):
    set_seed()
    model = depth_net(depth, residual).to(device).to(memory_format=torch.channels_last)
    opt = MySGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    spe = math.ceil(len(y_tr) / 128)
    return fit(model, opt, n_epochs, X_tr, y_tr, bs=128, aug=flip_crop,
               sched=cosine_warmup(opt, n_epochs * spe, warmup_steps=spe),
               tag=f"{'res' if residual else 'plain'}-{depth}")


R3b = {(d, r): cached(f"task3b_{'res' if r else 'plain'}{d}", lambda d=d, r=r: run_3b(d, r, epochs(20)))
       for d in (8, 20, 56) for r in (False, True)}

# %%
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
dcol = {8: "tab:green", 20: "tab:blue", 56: "tab:red"}
for (d, r), H in R3b.items():
    ep = np.arange(1, len(H["train_loss"]) + 1)
    style = dict(color=dcol[d], ls="-" if r else "--", label=f"{'ResNet' if r else 'plain'}-{d}")
    ax[0].plot(ep, H["train_loss"], marker="o", ms=3, **style)
    ax[1].plot(smooth(H["iter_loss"], 100), lw=1, **style)
    ax[2].plot(ep, H["val_acc"], marker="o", ms=3, **style)
ax[0].set(xlabel="epoch", ylabel="training loss (epoch mean)", title="Training loss vs epoch", yscale="log")
ax[1].set(xlabel="iteration", ylabel="training loss (moving avg, 100 it)", title="Training loss vs iteration",
          yscale="log")
ax[2].set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy")
for a in ax:
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
plt.tight_layout(); savefig("task3b_plain_vs_residual"); plt.show()

print(f"{'depth':>6s}{'plain loss':>12s}{'res loss':>10s}{'plain val':>11s}{'res val':>9s}   (final epoch)")
for d in (8, 20, 56):
    p, r = R3b[(d, False)], R3b[(d, True)]
    print(f"{d:6d}{p['train_loss'][-1]:12.4f}{r['train_loss'][-1]:10.4f}{p['val_acc'][-1]:11.4f}{r['val_acc'][-1]:9.4f}")

# %% [markdown]
# **Explanation.** _(filled in after running)_

# %% [markdown]
# ---
# ## Task 4: Best CNN with ≤ 500k parameters and ≤ 20 epochs
#
# **Design.** With only 20 epochs, a shallow-but-wide residual network trains faster than a deep-thin one
# (Task 3b), so the model is a ResNet with 2 basic blocks per stage (14 weighted layers), widths 26/52/104,
# projection (1×1 conv) shortcuts at the stage transitions and the last BN of every residual branch
# initialized to zero (each block starts as the identity). Recipe of the full model:
#
# * SGD with Nesterov momentum 0.9, batch 128, peak LR 0.1, 1 epoch linear warm-up then cosine decay to 0,
# * weight decay $5\times10^{-4}$, label smoothing 0.1,
# * augmentation: random flip + crop and cutout (16×16),
# * mixed precision, 20 epochs on the 45k training split; model selection only on the 5k validation split.
#
# **Ablation.** Each variant changes one design choice of the full model and is trained with the same seed
# and budget. The final model is the variant with the best validation accuracy; only that model is
# evaluated on the test set, once.

# %%
T4_ARCH = dict(blocks=(2, 2, 2), widths=(26, 52, 104), shortcut="B", zero_init_last_bn=True)
T4_PARAMS = n_params(CifarNet(**T4_ARCH))
print(f"Task 4 model parameters: {T4_PARAMS:,}")
assert T4_PARAMS <= 500_000

VARIANTS_4 = {
    "full model": dict(),
    "no shortcut (plain)": dict(residual=False),
    "no augmentation": dict(aug=None),
    "no cutout (flip+crop only)": dict(aug=flip_crop),
    "no weight decay": dict(wd=0.0),
    "no label smoothing": dict(ls=0.0),
    "AdamW instead of SGD": dict(opt="adamw"),
}


def run_4(name, n_epochs=20):
    cfg = VARIANTS_4[name]
    set_seed()
    model = CifarNet(**{**T4_ARCH, "residual": cfg.get("residual", True)}).to(device)
    model = model.to(memory_format=torch.channels_last)
    wd = cfg.get("wd", 5e-4)
    if cfg.get("opt") == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-2)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=wd)
    spe = math.ceil(len(y_tr) / 128)
    H = fit(model, opt, n_epochs, X_tr, y_tr, bs=128, aug=cfg.get("aug", compose(flip_crop, cutout)),
            label_smoothing=cfg.get("ls", 0.1), sched=cosine_warmup(opt, n_epochs * spe, warmup_steps=spe),
            tag=name)
    H["params"] = n_params(model)
    torch.save(model.state_dict(), f"{CKPT}/task4_{name.replace(' ', '_')}.pt")
    return H


R4 = {k: cached(f"task4_{k.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}",
                lambda k=k: run_4(k, epochs(20))) for k in VARIANTS_4}

# %%
print(f"{'variant':28s}{'params':>9s}{'final val acc':>15s}{'vs full':>11s}{'final train loss':>18s}")
base = R4["full model"]["val_acc"][-1]
for k, H in R4.items():
    print(f"{k:28s}{H['params']:9,d}{H['val_acc'][-1]:15.4f}{H['val_acc'][-1] - base:+11.4f}"
          f"{H['train_loss'][-1]:18.4f}")

fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
for i, (k, H) in enumerate(R4.items()):
    ep = np.arange(1, len(H["val_acc"]) + 1)
    lw = 2.5 if k == "full model" else 1.2
    ax[0].plot(ep, H["val_acc"], color=cmap(i), lw=lw, label=k)
    ax[1].plot(ep, H["train_loss"], color=cmap(i), lw=lw, label=k)
ax[0].set(xlabel="epoch", ylabel="accuracy", title="Task 4 ablation: validation accuracy", ylim=(0.5, 0.97))
ax[1].set(xlabel="epoch", ylabel="training loss", title="Task 4 ablation: training loss", yscale="log")
for a in ax:
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
plt.tight_layout(); savefig("task4_ablation"); plt.show()

# %% [markdown]
# **Ablation discussion.** _(filled in after running)_

# %% [markdown]
# ### Final model: single evaluation on the test set
#
# The variant with the highest final validation accuracy is selected and its saved weights are evaluated on
# the 10k CIFAR-10 test images. The result is cached, so the test set is used exactly once.

# %%
FINAL = max(R4, key=lambda k: R4[k]["val_acc"][-1])
print("selected on validation:", FINAL, f"(val acc {R4[FINAL]['val_acc'][-1]:.4f})")


def test_once():
    cfg = VARIANTS_4[FINAL]
    model = CifarNet(**{**T4_ARCH, "residual": cfg.get("residual", True)}).to(device)
    model.load_state_dict(torch.load(f"{CKPT}/task4_{FINAL.replace(' ', '_')}.pt", map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y_te), 1000):
            with torch.autocast(device.type, dtype=torch.float16, enabled=AMP):
                preds.append(model(X_te[i:i + 1000].to(device)).argmax(1).cpu())
    preds = torch.cat(preds)
    cm = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y_te, preds):
        cm[t, p] += 1
    return dict(model=FINAL, test_acc=(preds == y_te).float().mean().item(), confusion=cm.tolist())


T4_TEST = cached("task4_test", test_once)
print(f"Final model: {T4_TEST['model']}  |  parameters {T4_PARAMS:,}  |  TEST ACCURACY = {T4_TEST['test_acc']:.4f}")

cm = np.array(T4_TEST["confusion"])
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
im = ax[0].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(10):
    for j in range(10):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if i == j else "black")
ax[0].set_xticks(range(10)); ax[0].set_xticklabels(CLASSES, rotation=45); ax[0].set_yticks(range(10))
ax[0].set_yticklabels(CLASSES); ax[0].set(xlabel="predicted", ylabel="true", title="Test confusion matrix")
per_class = cm.diagonal() / cm.sum(1)
ax[1].barh(CLASSES, per_class, color="tab:blue"); ax[1].set_xlim(0.7, 1.0)
for i, v in enumerate(per_class):
    ax[1].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
ax[1].set(title="Per-class test accuracy", xlabel="accuracy"); ax[1].invert_yaxis()
plt.tight_layout(); savefig("task4_test"); plt.show()

# %% [markdown]
# **Test result.** _(filled in after running)_
#
# ---
# ## Summary
# _(filled in after running)_
