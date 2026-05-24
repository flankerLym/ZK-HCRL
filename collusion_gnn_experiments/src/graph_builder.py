from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "dev_mean", "dev_std", "lat_mean", "lat_std", "fail_rate", "rep_mean", "response_std", "dev_slope",
    "svc0", "svc1", "svc2",
]


def _safe_corr_matrix(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return np.eye(1)
    # rows = nodes, cols = time.
    mat = mat - mat.mean(axis=1, keepdims=True)
    denom = np.sqrt((mat ** 2).sum(axis=1, keepdims=True))
    norm = mat / np.maximum(denom, 1e-8)
    corr = norm @ norm.T
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def build_window_graph(panel: pd.DataFrame, window_id: int, start: int, end: int, n_oracles: int, top_k: int = 8):
    w = panel[(panel["step"] >= start) & (panel["step"] < end)].copy()
    if w.empty:
        raise ValueError("Empty window")

    # Node-level features.
    rows = []
    response_mat = []
    dev_mat = []
    fail_mat = []
    lat_mat = []
    for oid in range(n_oracles):
        g = w[w["oracle_id"] == oid].sort_values("step")
        if g.empty:
            vals = np.zeros(10)
            svc = oid % 3
            label = 0
            resp = np.zeros(end - start)
            dev = np.zeros(end - start)
            fail = np.zeros(end - start)
            lat = np.zeros(end - start)
        else:
            dev_values = g["deviation"].to_numpy(float)
            lat_values = g["latency"].to_numpy(float)
            fail_values = g["validation_fail"].to_numpy(float)
            rep_values = g["reputation"].to_numpy(float)
            resp_values = g["response_norm"].to_numpy(float)
            svc = int(g["service_type"].iloc[-1])
            label = int(g["is_colluder"].iloc[-1])
            if len(dev_values) >= 2:
                dev_slope = float(np.polyfit(np.arange(len(dev_values)), dev_values, deg=1)[0])
            else:
                dev_slope = 0.0
            vals = np.array([
                dev_values.mean(), dev_values.std(), lat_values.mean() / 1000.0, lat_values.std() / 1000.0,
                fail_values.mean(), rep_values.mean(), resp_values.std(), dev_slope,
                1.0 if svc == 0 else 0.0, 1.0 if svc == 1 else 0.0, 1.0 if svc == 2 else 0.0,
            ], dtype=float)
            resp = resp_values
            dev = dev_values
            fail = fail_values
            lat = lat_values / max(lat_values.mean(), 1e-6)
        rows.append({"oracle_id": oid, "window_id": window_id, "label": label, **{c: vals[j] for j, c in enumerate(FEATURE_COLUMNS)}})
        response_mat.append(_pad(resp, end - start))
        dev_mat.append(_pad(dev, end - start))
        fail_mat.append(_pad(fail, end - start))
        lat_mat.append(_pad(lat, end - start))

    node_df = pd.DataFrame(rows)
    X = node_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = node_df["label"].to_numpy(dtype=int)

    response_corr = np.maximum(_safe_corr_matrix(np.vstack(response_mat)), 0.0)
    dev_corr = np.maximum(_safe_corr_matrix(np.vstack(dev_mat)), 0.0)
    fail_corr = np.maximum(_safe_corr_matrix(np.vstack(fail_mat)), 0.0)
    lat_corr = np.maximum(_safe_corr_matrix(np.vstack(lat_mat)), 0.0)

    A = 0.35 * response_corr + 0.25 * dev_corr + 0.25 * fail_corr + 0.15 * lat_corr
    np.fill_diagonal(A, 1.0)

    # Keep top-k neighbors for each node to avoid a dense noisy graph.
    if top_k is not None and top_k > 0 and top_k < n_oracles:
        B = np.zeros_like(A)
        for i in range(n_oracles):
            idx = np.argsort(-A[i])[:top_k + 1]
            B[i, idx] = A[i, idx]
        A = np.maximum(B, B.T)
        np.fill_diagonal(A, 1.0)

    return X, y, A, node_df


def _pad(v, length):
    v = np.asarray(v, dtype=float)
    if len(v) >= length:
        return v[:length]
    if len(v) == 0:
        return np.zeros(length)
    return np.pad(v, (0, length - len(v)), mode="edge")


def build_graph_dataset(panel: pd.DataFrame, n_oracles: int, window_size: int, top_k: int = 8):
    max_step = int(panel["step"].max()) + 1
    windows = []
    node_frames = []
    for wid, start in enumerate(range(0, max_step, window_size)):
        end = min(start + window_size, max_step)
        if end - start < max(10, window_size // 4):
            continue
        X, y, A, node_df = build_window_graph(panel, wid, start, end, n_oracles, top_k=top_k)
        phase = panel[(panel["step"] >= start) & (panel["step"] < end)]["phase"].mode().iloc[0]
        windows.append({"window_id": wid, "start": start, "end": end, "phase": phase, "X": X, "y": y, "A": A})
        node_df["phase"] = phase
        node_df["start"] = start
        node_df["end"] = end
        node_frames.append(node_df)
    return windows, pd.concat(node_frames, ignore_index=True)


def graph_diagnostics(A: np.ndarray, y: np.ndarray):
    y = np.asarray(y).astype(int)
    coll = y == 1
    ben = y == 0
    if coll.sum() <= 1:
        cc = np.nan
    else:
        cc = A[np.ix_(coll, coll)][~np.eye(coll.sum(), dtype=bool)].mean()
    if ben.sum() <= 1:
        bb = np.nan
    else:
        bb = A[np.ix_(ben, ben)][~np.eye(ben.sum(), dtype=bool)].mean()
    cb = A[np.ix_(coll, ben)].mean() if coll.any() and ben.any() else np.nan
    return {"colluder_colluder_weight": cc, "benign_benign_weight": bb, "colluder_benign_weight": cb, "collusion_graph_gap": cc - cb if np.isfinite(cc) and np.isfinite(cb) else np.nan}
