#!/usr/bin/env python3
"""Run me: `python examples/quickstart.py`

Uses the bundled vote matrix — ten real LLM voters over 80 items, no network, no API key —
to show three things in about a second:

  1. On the held-out half, the independence-weighted vote is right more often than the plain
     majority, and its confidence is less overstated; on 29 items that is what happened here,
     not a rule.
  2. Agreement is not accuracy: the plain majority's confidence runs well above how often it is
     right.
  3. On a panel that is one model throughout, weighting by provenance does exactly nothing.

The run itself lives in `voteworth.demo`, so that `python -m voteworth.demo`
after a plain `pip install` shows the identical output. One implementation, two entry points —
a second copy here would drift.

Data: 10.5281/zenodo.21773065 (CC BY 4.0). Measurement note: 10.5281/zenodo.21775275.
"""
from voteworth.demo import main

if __name__ == "__main__":
    main()
