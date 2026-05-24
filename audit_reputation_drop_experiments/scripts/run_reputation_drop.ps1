python .\audit_reputation_drop_experiments\run_audit_reputation_drop.py `
  --trace .\experiments_real_trace\data\real_oracle_trace.csv `
  --out .\audit_reputation_drop_experiments\output `
  --seeds 3,4,5,6,7 `
  --requests 6000 `
  --oracles 120 `
  --malicious-ratio 0.30
python .\audit_reputation_drop_experiments\plot_reputation_curves.py `
  --input .\audit_reputation_drop_experiments\output\audit_reputation_curve.csv `
  --out .\audit_reputation_drop_experiments\output\plots
