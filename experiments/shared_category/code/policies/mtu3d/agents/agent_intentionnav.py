"""Canonical entry point for the IntentionNav reference agent.

The implementation remains in :mod:`agent_vlm_engine` for compatibility with
existing experiment scripts and released logs.  New experiments should use
this name.

Examples:
  # Implicit intent, hosted VLM for intent inference and sparse visual cues
  python agents/agent_intentionnav.py --model gemini_3_1_flash --style formal

  # No-API navigation-only calibration with the explicit target category
  python agents/agent_intentionnav.py --objectnav --local-perception-only \
      --style formal --limit 10
"""
from agent_vlm_engine import main


if __name__ == "__main__":
    main()
