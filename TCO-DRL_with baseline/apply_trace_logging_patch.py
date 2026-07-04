"""Apply HCRL trace logging patch to main.py.

Run this script from `TCO-DRL_with baseline/` after copying the patch files:
    python apply_trace_logging_patch.py

It is idempotent: running it multiple times will not duplicate inserts.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.py"
BACKUP = ROOT / "main.py.before_hcrl_trace_patch"

IMPORT_LINE = "from trace_export_utils import HCRLTraceExporter\n"
INIT_MARK = "# [HCRL_TRACE_EXPORT] initialize exporter"
DECISION_MARK = "# [HCRL_TRACE_EXPORT] capture decision evidence before environment transition"
EXEC_MARK = "# [HCRL_TRACE_EXPORT] capture execution/audit evidence after environment transition"
SAVE_MARK = "# [HCRL_TRACE_EXPORT] save per-request traces"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Cannot find target block:\n{old[:300]}")
    return text.replace(old, new, 1)


def main() -> None:
    if not MAIN.exists():
        raise FileNotFoundError(f"main.py not found: {MAIN}")
    text = MAIN.read_text(encoding="utf-8")
    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")
        print(f"Backup saved: {BACKUP}")

    if "HCRLTraceExporter" not in text:
        text = replace_once(text, "from utils import get_args\n", "from utils import get_args\n" + IMPORT_LINE)

    if INIT_MARK not in text:
        text = replace_once(
            text,
            "last_results = None\n",
            "last_results = None\n\n"
            f"{INIT_MARK}\n"
            "trace_exporter = HCRLTraceExporter(RUN_DIR, RUN_ID, args)\n"
            "if trace_exporter.enabled:\n"
            "    print(f\"[HCRL trace] enabled; trace files will be saved under {RUN_DIR}\")\n",
        )

    target_feedback = '            feedback = env.feedback_hcrl(request_attrs, mode_action, primary_action, backup_action, "HCRL-Oracle")\n'
    if DECISION_MARK not in text:
        decision_block = (
            f"\n            {DECISION_MARK}\n"
            "            trace_exporter.capture_decision(\n"
            "                env, args, RUN_ID, episode, request_attrs,\n"
            "                mode_action, primary_action, backup_action, time_period, global_step\n"
            "            )\n"
        )
        text = replace_once(text, target_feedback, decision_block + target_feedback)

    if EXEC_MARK not in text:
        exec_block = (
            target_feedback +
            f"            {EXEC_MARK}\n"
            "            trace_exporter.capture_execution(\n"
            "                env, args, RUN_ID, episode, request_attrs,\n"
            "                mode_action, primary_action, backup_action, feedback\n"
            "            )\n"
        )
        text = replace_once(text, target_feedback, exec_block)

    if SAVE_MARK not in text:
        text = replace_once(
            text,
            "# Save final results.\n",
            f"{SAVE_MARK}\n"
            "if trace_exporter is not None and trace_exporter.enabled:\n"
            "    trace_exporter.save()\n\n"
            "# Save final results.\n",
        )

    MAIN.write_text(text, encoding="utf-8")
    print(f"Patched successfully: {MAIN}")
    print("New outputs during normal HCRL training:")
    print("  *_hcrl_zk_schedule_trace.csv")
    print("  *_hcrl_execution_trace.csv")
    print("  *_hcrl_audit_trace.csv")
    print("  *_oracle_pool_snapshot.jsonl")
    print("  *_trace_manifest.json")


if __name__ == "__main__":
    main()
