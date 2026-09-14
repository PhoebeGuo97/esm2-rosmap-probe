# Does a protein language model encode AD/PD co-dysregulation?

A sequence-only test of whether ESM-2 representations separate the 189 proteins
found concordantly dysregulated in Alzheimer's and Parkinson's disease brain
(ROSMAP, n = 125) from background human proteins.

The point of the experiment is the evaluation, not the accuracy. A protein
language model sees only amino acid sequence, and disease dysregulation is a
property of cellular context rather than of sequence, so the honest prior is
that this should work poorly. What the experiment establishes is how poorly,
with calibrated uncertainty, against a null control.

## Design

| | |
|---|---|
| Positives | 189 unique UniProt entries from the concordant AD/PD protein set |
| Negatives | length-matched sample of the reviewed human proteome (3:1), or a user-supplied set |
| Representation | mean-pooled final-layer ESM-2 (`facebook/esm2_t12_35M_UR50D`) |
| Models | L2 logistic regression on frozen embeddings; ESM-2 with top 2 encoder layers unfrozen |
| Validation | 5-fold stratified CV, pooled out-of-fold predictions |
| Discrimination | AUROC, AUPRC, with 2000-sample bootstrap 95% CI over proteins |
| Calibration | Brier score, 15-bin expected calibration error |
| Null control | label permutation, expected AUROC near 0.500 |

Sequence length is the strongest confounder available to a sequence-only
classifier, so the background is length-matched by decile rather than sampled
uniformly. Class imbalance is handled in the loss, not by resampling.

## Run it

```bash
pip install -r requirements.txt
python run_experiment.py                # ~20-40 min on an M-series Mac
python run_experiment.py --quick        # ~5 min, 8M model, 3 folds, smoke test
```

The first run downloads the reviewed human proteome from UniProt (about 30 MB)
and caches it as `uniprot_human_reviewed.tsv`.

### The background set matters

By default the negatives are drawn from the whole reviewed human proteome. That
makes the comparison easy in an uninteresting way: a classifier can succeed by
detecting general neuronal or mitochondrial sequence character rather than
anything about disease. The interpretable version uses the proteins that were
quantified in the same ROSMAP dataset but were **not** concordant:

```bash
python run_experiment.py --background rosmap_quantified_not_concordant.txt
```

One gene symbol per line. Any symbol UniProt cannot resolve is written to
`unmatched_genes.txt` and reported rather than silently dropped.

## Outputs

- `results.json` — metrics, CIs, config, and run metadata
- `predictions.csv` — per-protein out-of-fold probabilities
- `figures/roc_pr.png`, `figures/calibration.png`
- `unmatched_genes.txt`

## Interpreting the result

- **Null control near 0.500** confirms the CV and metric code are sound. Under
  this design the null has a standard deviation of roughly 0.04, so a real
  AUROC below about 0.58 is not distinguishable from chance at n = 756.
- **A modest AUROC with poor calibration** is the expected and informative
  outcome: the representation carries some signal (protein family, subcellular
  compartment, abundance-correlated composition) but the predicted
  probabilities should not be read as confidence.
- **A high AUROC against the random-proteome background** is a reason to
  suspect the confound, not a result. Re-run with the ROSMAP background before
  believing it.
