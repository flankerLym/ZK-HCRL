# Collusion-aware GNN Experiments for HCRL-Oracle

This folder is a standalone experiment package to **single out the HCRL oracle graph/GNN idea** and test whether a behavior-correlation graph helps detect collusive oracle risk.

It is designed to be copied/extracted into the root of:

```text
TCO-DRL/
```

It does **not** overwrite your original baseline or `experiments_real_trace` folder.

## What this experiment tests

**No PyTorch is required.** The graph-aware encoder is implemented with pure NumPy message passing over the oracle behavior graph, matching the lightweight style of the original TCO-DRL codebase.


Ordinary malicious-ratio experiments treat malicious oracles as independent. This experiment creates dynamic collusion scenarios where multiple oracles show correlated abnormal behavior:

1. `coordinated_shift`: colluding oracles jointly shift response values.
2. `reputation_poisoning_collusion`: colluding oracles behave well first, then attack after gaining high reputation.
3. `latency_copattern`: colluding oracles coordinate delayed responses.
4. `failure_cooccurrence`: colluding oracles fail together in the same time windows.
5. `intermittent_evasion`: colluding oracles attack intermittently to evade audit.
6. `gradual_drift`: colluding oracles slowly drift away from the reference value.

The experiment builds an oracle behavior graph using:

- response correlation
- deviation similarity
- failure co-occurrence
- latency co-pattern

Then it compares:

- `Feature-MLP`: NumPy feature-only risk scorer, no graph
- `Collusion-GNN`: NumPy graph-aware oracle encoder
- `Heuristic-Risk`: simple hand-crafted risk score

The goal is to support a paper claim like:

> A collusion-aware oracle graph enables the scheduler to capture group-level oracle risk that cannot be represented by independent malicious labels alone.

## Quick start: PowerShell

From the `TCO-DRL` root:

```powershell
.\collusion_gnn_experiments\run_collusion_gnn_experiment.ps1
```

This will use:

```text
.\experiments_real_trace\data\real_oracle_trace.csv
```

and write results to:

```text
.\collusion_gnn_experiments\output
```

## Manual command

```powershell
python .\collusion_gnn_experiments\run_collusion_gnn_experiments.py `
  --trace .\experiments_real_trace\data\real_oracle_trace.csv `
  --out .\collusion_gnn_experiments\output `
  --seeds 3,4,5,6,7 `
  --requests 12000 `
  --oracles 120 `
  --malicious-ratio 0.30 `
  --window-size 120 `
  --epochs 80
```

Then plot:

```powershell
python .\collusion_gnn_experiments\plot_collusion_gnn_results.py `
  --input .\collusion_gnn_experiments\output `
  --out .\collusion_gnn_experiments\output\figures
```

## Main outputs

```text
output/collusion_gnn_metrics_by_seed.csv
output/collusion_gnn_summary_mean_std.csv
output/collusion_gnn_window_risk.csv
output/collusion_graph_diagnostics.csv
output/paper_table_collusion_gnn.csv
output/paper_table_collusion_gnn.tex
output/figures/collusion_gnn_performance.png
output/figures/collusion_risk_curves_all_scenarios.png
output/figures/collusion_graph_heatmap.png
```

## How to use in the paper

Suggested section title:

```text
Collusion-aware Oracle Risk Modeling
```

Suggested claim:

```text
Compared with feature-only risk scoring, the collusion-aware graph encoder improves detection of coordinated oracle attacks by exploiting behavioral correlations among oracle nodes, including response similarity, failure co-occurrence, and latency co-patterns.
```

