#!/usr/bin/env python3
"""Run me: `python examples/quickstart.py`

Uses the bundled vote matrix — ten real LLM voters over 80 items, no network, no API key —
to show three things in about a second:

  1. A plain majority over a panel with a correlated bloc scores WORSE than the panel's best
     single voter.
  2. Independence weighting recovers part of the gap and lowers the overstated confidence.
  3. On a panel that is one model throughout, the same weighting does exactly nothing.

The run itself lives in `voteworth.demo`, so that `python -m voteworth.demo`
after a plain `pip install` shows the identical output. One implementation, two entry points —
a second copy here would drift.

Data: 10.5281/zenodo.21773065 (CC BY 4.0). Measurement note: 10.5281/zenodo.21775275.
"""
from voteworth.demo import main

if __name__ == "__main__":
    main()
