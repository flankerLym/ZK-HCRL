#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the ZK-VOS pressure test tool.

The canonical implementation lives under:
  zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py

This wrapper allows the older command style:
  python "TCO-DRL_with baseline/tools/zk_vos_pressure_test.py" ...
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
TARGET = REPO_ROOT / "zk_vos_real_circom_hcrl_patch" / "experiments" / "zk_vos" / "tools" / "zk_vos_pressure_test.py"

if not TARGET.exists():
    raise SystemExit(f"Cannot find canonical ZK-VOS pressure test tool: {TARGET}")

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
