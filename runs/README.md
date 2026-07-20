# Run artifacts (provenance)

Every number in the report, the README table, and `results/figures/` is derived from
the files in this directory — published so the results are independently auditable
rather than only visible as rendered outputs.

## What is here (per run)

| File | Purpose |
|---|---|
| `config.yaml` | the exact config the run executed with (copied by `src/train.py` at start-up) |
| `metrics.json` | dev/eval EER, generalization gap, min t-DCF, AUC, accuracy, trainable-parameter count, per-attack EER |
| `log.csv` | per-epoch train loss, dev EER, learning rate, wall-clock seconds |
| `scores_dev.npz`, `scores_eval.npz` | raw per-utterance detector scores + labels + attack IDs |
| `tsne_eval.png` | embedding visualization (where produced) |

Model checkpoints (`best.pt`, `last.pt`) are **not** published: the SSL checkpoints
store the frozen 95 M-parameter backbone and are too large for a code repository.
Everything needed to *verify the reported numbers* is here without them.

## Recomputing the reported results

From the raw scores, without a GPU and without the 23.6 GB dataset:

```bash
python scripts/collate_results.py   # -> results/RESULTS.md  (the report's main table)
python scripts/make_figures.py      # -> gap_bars, per_attack, efficiency, training_curves
python scripts/make_roc_fig.py      # -> roc_scores.png (recomputes ROC/AUC from scores)
```

EER, min t-DCF, AUC and the per-attack breakdown can all be recomputed directly from
`scores_eval.npz` with `src/metrics.py` (min t-DCF additionally needs the ASV score
file shipped inside the dataset).

## Environment the reported runs used

| | |
|---|---|
| Python | 3.10 |
| torch / torchaudio | 2.5.1 (+cu121) |
| transformers | 4.49.0 |
| peft | 0.15.2 |
| numpy | 2.2.6 |
| LCNN runs | local NVIDIA RTX 3050 Ti (4 GB) |
| wav2vec2 runs | NVIDIA L4 (Google Colab) |
| seed | 1234 (all runs) |

Pinned in `requirements.txt` / `environment.yml`. Each run was uninterrupted (no
checkpoint resumes), which matters because our checkpoints do not store RNG state —
see the reproducibility note in the report.

## Honest gaps

- **`baseline_lcnn_lfcc/` has no `scores_*.npz`.** The score arrays for this run were
  lost to an accidental cleanup after the run completed. Its `metrics.json` holds the
  real measured values (recorded from the completed run, flagged via a `_source`
  field), and its per-attack breakdown is intact — but unlike the other runs, its EER
  cannot be *recomputed from scratch* here. Re-running
  `python scripts/run_all.py --only baseline_lcnn_lfcc` regenerates them (~20 min on a
  small GPU).
- **`wav2vec2_lora/metrics.json`** was recomputed from that run's `scores_eval.npz`
  using `src/metrics.py`, so its eval numbers are reproducible from the published
  scores; its `dev_eer_seen` (0.16%) is taken from the training log.
- `log.csv` is absent for the runs executed on Colab where only the final artifacts
  were retrieved.

Files carrying a `_source` field in `metrics.json` were recorded or recomputed rather
than written directly by a live `src/eval.py` process; the field says which.
