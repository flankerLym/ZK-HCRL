from __future__ import annotations
import numpy as np


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _normalize_adj_np(A: np.ndarray) -> np.ndarray:
    """Symmetric adjacency normalization used by the graph encoder.

    This is a pure NumPy version of the GCN-style operation D^{-1/2} A D^{-1/2}.
    It keeps this standalone experiment consistent with a graph/GNN-style oracle encoder
    without requiring PyTorch or torch-geometric.
    """
    A = np.asarray(A, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    deg = A.sum(axis=1)
    inv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    return (A * inv[:, None]) * inv[None, :]


class FeatureMLP:
    """Feature-only oracle risk scorer implemented as NumPy logistic regression.

    The old package used a small PyTorch MLP. Your original TCO-DRL/HCRL codebase does
    not depend on torch, so this replacement keeps the same public name used by the
    runner while training a lightweight feature-only baseline with NumPy only.
    """

    method_name = "Feature-MLP"

    def __init__(self, in_dim: int, hidden: int = 32):
        self.in_dim = int(in_dim)
        self.hidden = hidden
        self.use_graph = False
        self.mu = None
        self.sd = None
        self.w = None
        self.b = 0.0

    def make_features(self, wdict):
        X = np.asarray(wdict["X"], dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class CollusionGCN:
    """Pure NumPy collusion-aware graph encoder.

    This class isolates the HCRL-style oracle graph encoding idea without introducing
    PyTorch. It builds graph-aware node features by message passing over the behavior
    correlation graph:

        H0 = X
        H1 = A_hat X
        H2 = A_hat^2 X

    The final risk scorer is a logistic classifier on [H0, H1, H2, graph diagnostics].
    This is intentionally lightweight: the experiment tests whether graph structure
    helps reveal collusive risk, not whether a deep neural implementation is necessary.
    """

    method_name = "Collusion-GNN"

    def __init__(self, in_dim: int, hidden: int = 32):
        self.in_dim = int(in_dim)
        self.hidden = hidden
        self.use_graph = True
        self.mu = None
        self.sd = None
        self.w = None
        self.b = 0.0

    def make_features(self, wdict):
        X = np.asarray(wdict["X"], dtype=float)
        A = np.asarray(wdict["A"], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Ahat = _normalize_adj_np(A)
        H1 = Ahat @ X
        H2 = Ahat @ H1
        degree = A.sum(axis=1, keepdims=True)
        degree = degree / max(A.shape[0], 1)
        offdiag = A.copy()
        np.fill_diagonal(offdiag, 0.0)
        max_neighbor = offdiag.max(axis=1, keepdims=True)
        mean_neighbor = offdiag.mean(axis=1, keepdims=True)
        graph_stats = np.concatenate([degree, max_neighbor, mean_neighbor], axis=1)
        return np.concatenate([X, H1, H2, graph_stats], axis=1)


def _stack_dataset(windows, model):
    Xs, ys, wids = [], [], []
    for wdict in windows:
        F = model.make_features(wdict)
        y = np.asarray(wdict["y"], dtype=int)
        Xs.append(F)
        ys.append(y)
        wids.append(np.full(len(y), int(wdict["window_id"]), dtype=int))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(wids)


def _standardize_train_test(train_X, test_X):
    mu = train_X.mean(axis=0, keepdims=True)
    sd = train_X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (train_X - mu) / sd, (test_X - mu) / sd, mu, sd


def train_model(model, train_windows, test_windows, epochs=80, lr=0.08, weight_decay=1e-4, seed=0, verbose=False):
    """Train the feature-only or graph-aware risk scorer with NumPy.

    The function keeps the same name/signature as the previous torch version, so the
    existing runner script does not need to change. It uses weighted binary logistic
    regression optimized by full-batch gradient descent.
    """
    rng = np.random.default_rng(seed)
    train_X, train_y, _ = _stack_dataset(train_windows, model)
    test_X, test_y, test_wids = _stack_dataset(test_windows, model)
    train_X, test_X, mu, sd = _standardize_train_test(train_X, test_X)
    model.mu, model.sd = mu, sd

    n, d = train_X.shape
    model.w = rng.normal(0.0, 0.02, size=d)
    model.b = 0.0

    # Class-balanced sample weights.
    pos = max(float((train_y == 1).sum()), 1.0)
    neg = max(float((train_y == 0).sum()), 1.0)
    sample_w = np.where(train_y == 1, neg / pos, 1.0).astype(float)
    sample_w = sample_w / max(sample_w.mean(), 1e-8)

    # A slightly decayed learning rate makes the output stable across Windows/Python versions.
    for ep in range(int(max(1, epochs))):
        logits = train_X @ model.w + model.b
        p = _sigmoid(logits)
        err = (p - train_y) * sample_w
        grad_w = (train_X.T @ err) / n + weight_decay * model.w
        grad_b = float(err.mean())
        step = lr / np.sqrt(1.0 + 0.03 * ep)
        model.w -= step * grad_w
        model.b -= step * grad_b
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            eps = 1e-8
            loss = -np.mean(sample_w * (train_y * np.log(p + eps) + (1 - train_y) * np.log(1 - p + eps)))
            print(f"epoch={ep:03d} loss={loss:.4f}")

    scores = _sigmoid(test_X @ model.w + model.b)
    return test_y, scores, test_wids


def predict_model(model, windows):
    X, y, wids = _stack_dataset(windows, model)
    if model.mu is not None and model.sd is not None:
        X = (X - model.mu) / model.sd
    scores = _sigmoid(X @ model.w + model.b)
    return y, scores, wids


def heuristic_risk_scores(windows):
    scores, labels, wids = [], [], []
    for wdict in windows:
        X = np.asarray(wdict["X"], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # X columns: dev_mean, dev_std, lat_mean, lat_std, fail_rate, rep_mean,
        # response_std, dev_slope, service_type one-hot...
        raw = 1.8 * X[:, 0] + 1.2 * X[:, 4] + 0.8 * X[:, 2] + 0.7 * X[:, 6] - 0.9 * X[:, 5]
        raw = (raw - np.nanmean(raw)) / max(np.nanstd(raw), 1e-6)
        s = _sigmoid(raw)
        scores.append(s)
        labels.append(wdict["y"])
        wids.append(np.full(len(wdict["y"]), wdict["window_id"]))
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(wids)
