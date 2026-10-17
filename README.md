# AERO-Lite

A deployable demo of **AERO (Autonomous Edge-Resilience Orchestrator)** — the
predictive SDN failover system from my final-year capstone. The full project
(github.com/Darchis01/aero) uses Mininet and a Ryu SDN controller to run a
real virtual network simulation on Linux. This version replaces that
Linux-dependent simulation layer with a synthetic traffic generator, so the
same ML-driven prediction logic can run anywhere — including free serverless
hosting — with no VM, no root access, and no persistent server process.

**Live demo:** _add your Vercel URL here after deploying_

## What's real here, and what's synthetic

| Component | Full AERO | AERO-Lite |
|---|---|---|
| Network topology | Mininet virtual switches/hosts | Fixed topology, defined as data |
| Traffic / link stats | Ryu reading live OpenFlow port counters | Synthetic generator producing correlated, time-varying metrics in the same feature shape |
| ML model | Random Forest (scikit-learn) | **The same kind of model, retrained** — see below — and genuinely run at inference time |
| Failure injection | Manually blocking a real Mininet link | Flag sent from the dashboard, applied to the synthetic generator |
| Backend | Long-running Flask process | Stateless Vercel serverless function |

The prediction step is not faked. `training/train_model.py` trains a real
`RandomForestClassifier` on a larger, more realistic dataset than the
original project used (8,000 samples with correlated features and a soft
probabilistic failure boundary, instead of 1,200 independent-random samples
with a hard OR-threshold label — see the file for details). The trained
forest is then exported to `api/_lib/forest.json` and evaluated with a small
pure-Python tree-walker (`api/forest_predict.py`) instead of loading
scikit-learn at runtime. This was verified to produce **identical
predictions to the original sklearn model** across thousands of test
vectors — it's the same model, just without a heavy runtime dependency.

## Architecture

```
aero-demo/
├── api/
│   ├── simulate.py       Stateless serverless endpoint (POST /api/simulate)
│   ├── simulator.py      Synthetic topology + per-link traffic generation
│   ├── forest_predict.py Pure-Python Random Forest inference
│   ├── _lib/forest.json  Exported trained model (tree structure as data)
│   └── requirements.txt  Intentionally empty — stdlib only
├── public/
│   ├── index.html        Dashboard shell
│   ├── style.css
│   └── dashboard.js       Polls /api/simulate every 2.5s, renders topology/stats
├── training/
│   ├── train_model.py    Retrains the model and re-exports forest.json
│   └── requirements.txt  scikit-learn, numpy, joblib (offline use only)
└── vercel.json
```

The whole system is **stateless by design**: the dashboard keeps track of
the simulated "tick" and any manually injected failures, and sends that with
every request. The backend is a pure function of its input — no database, no
file writes, no background thread — which is what makes it safe to run as a
serverless function instead of needing an always-on process like the
original Flask + Ryu setup did.

## Running locally

```bash
pip install -r training/requirements.txt   # only if you want to retrain
npm install -g vercel
vercel dev
```

This serves the dashboard and the `/api/simulate` function locally, exactly
as Vercel would in production.

## Deploying

1. Push this folder to its own GitHub repo.
2. Go to [vercel.com](https://vercel.com), sign in with GitHub, import the repo.
3. No build configuration needed — Vercel auto-detects the static `public/`
   folder and the Python function under `api/`.
4. Deploy. You get a public URL on Vercel's free tier, redeployed
   automatically on every push to `main`.

## Retraining the model

```bash
cd training
pip install -r requirements.txt
python3 train_model.py
```

This prints cross-validated accuracy, a classification report, and feature
importances, then saves `model.pkl`. To update the deployed version, re-run
the export step (see the inline comments in `train_model.py` for the
tree-export snippet, or ask for it to be regenerated) to refresh
`api/_lib/forest.json`, then redeploy.

## License

All rights reserved. This code is provided publicly for demonstration and
portfolio purposes only. No permission is granted to copy, modify,
distribute, or reuse this code or any derivative work without explicit
written permission from the author.

Copyright (c) 2026 Chisom Darlynton Steve-Oko.
