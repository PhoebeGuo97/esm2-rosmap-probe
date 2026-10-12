#!/usr/bin/env python3
"""
Can a protein language model distinguish proteins concordantly dysregulated in
Alzheimer's and Parkinson's disease (ROSMAP brain proteomics) from
length-matched background human proteins, using sequence alone?

Two models are compared on identical cross-validation folds:
  1. frozen  : mean-pooled ESM-2 embeddings -> L2 logistic regression
  2. finetune: ESM-2 with the top N encoder layers unfrozen + classification head

Both are scored on discrimination (AUROC, AUPRC) and calibration
(Brier score, expected calibration error) from pooled out-of-fold predictions.
A label-permutation control is run on the frozen model as a negative control.

Usage
-----
  python run_experiment.py                      # full run, defaults below
  python run_experiment.py --quick              # 8M model, 2 epochs, 3 folds
  python run_experiment.py --background my.txt  # supply your own negative set
  python run_experiment.py --outdir results/x   # write outputs to a folder

Outputs (in --outdir, default the current directory)
-------
  results.json          all metrics, config, and run metadata
  predictions.csv       per-protein out-of-fold predicted probabilities
  figures/roc_pr.png    ROC and precision-recall curves
  figures/calibration.png
  unmatched_genes.txt   input symbols with no UniProt match
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

UNIPROT_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=%28organism_id%3A9606%29+AND+%28reviewed%3Atrue%29"
    "&fields=accession,gene_primary,gene_names,sequence,length,protein_name"
    "&format=tsv"
)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def download_proteome(cache: Path) -> pd.DataFrame:
    """Download (once) the reviewed human proteome with sequences."""
    if not cache.exists():
        import urllib.request

        print(f"Downloading reviewed human proteome from UniProt -> {cache.name}")
        print("(about 30 MB, one time only)")
        urllib.request.urlretrieve(UNIPROT_STREAM, cache)
    df = pd.read_csv(cache, sep="\t")
    df = df.rename(
        columns={
            "Entry": "accession",
            "Gene Names (primary)": "gene_primary",
            "Gene Names": "gene_names",
            "Sequence": "sequence",
            "Length": "length",
            "Protein names": "protein_name",
        }
    )
    df = df.dropna(subset=["sequence"])
    print(f"Loaded {len(df):,} reviewed human proteins")
    return df


def build_symbol_index(df: pd.DataFrame) -> dict:
    """Map every gene symbol and synonym to a row index, primary names winning."""
    index = {}
    for i, names in df["gene_names"].items():
        if not isinstance(names, str):
            continue
        for sym in names.split():
            index.setdefault(sym.upper(), i)
    for i, prim in df["gene_primary"].items():  # primary names override synonyms
        if isinstance(prim, str) and prim:
            index[prim.upper()] = i
    return index


def resolve(symbols, df, index):
    """Return (dataframe of matched proteins, list of unmatched symbols)."""
    rows, unmatched, seen = [], [], set()
    for sym in symbols:
        i = index.get(sym.upper())
        if i is None:
            unmatched.append(sym)
            continue
        acc = df.at[i, "accession"]
        if acc in seen:
            continue
        seen.add(acc)
        rows.append(i)
    return df.loc[rows].copy(), unmatched


def length_matched_background(pos, pool, ratio, rng):
    """Sample negatives whose length distribution matches the positives.

    Length is the single strongest confounder for a sequence-only classifier,
    so it is controlled by construction rather than hoped away.
    """
    pool = pool.copy()
    edges = np.quantile(pos["length"], np.linspace(0, 1, 11))
    edges[0], edges[-1] = 0, np.inf
    pos_bin = np.digitize(pos["length"], edges[1:-1])
    pool_bin = np.digitize(pool["length"], edges[1:-1])
    picked = []
    for b in range(11):
        want = int((pos_bin == b).sum() * ratio)
        cand = pool.index[pool_bin == b].tolist()
        if not want or not cand:
            continue
        rng.shuffle(cand)
        picked.extend(cand[:want])
    return pool.loc[picked].copy()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def expected_calibration_error(y, p, n_bins=15):
    """Equal-width-bin ECE, the standard formulation."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def score(y, p):
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(np.asarray(y), np.asarray(p)),
        "prevalence": float(np.mean(y)),
        "n": int(len(y)),
    }


def bootstrap_ci(y, p, fn, n_boot, rng, alpha=0.05):
    """Percentile CI over resampled proteins."""
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        stats.append(fn(y[i], p[i]))
    if not stats:
        return None
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed(seqs, model_name, device, max_len, batch_size):
    """Mean-pooled final-layer ESM-2 representations, masked over real residues."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            batch = seqs[s : s + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
            done = min(s + batch_size, len(seqs))
            print(f"  embedded {done}/{len(seqs)}", end="\r", flush=True)
    print(f"  embedded {len(seqs)}/{len(seqs)} in {time.time() - t0:.0f}s")
    del model
    return np.vstack(out)


def run_frozen(X, y, folds, seed):
    """Out-of-fold probabilities from L2 logistic regression on frozen embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def run_finetune(seqs, y, model_name, device, max_len, batch_size, epochs,
                 folds, unfreeze, lr, seed):
    """Out-of-fold probabilities from ESM-2 with the top `unfreeze` layers trained."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, s, lab):
            self.s, self.lab = s, lab

        def __len__(self):
            return len(self.s)

        def __getitem__(self, i):
            return self.s[i], self.lab[i]

    def collate(batch):
        s, lab = zip(*batch)
        enc = tok(
            list(s),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc["labels"] = torch.tensor(lab, dtype=torch.long)
        return enc

    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(seqs, y), 1):
        torch.manual_seed(seed + k)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)

        n_layers = model.config.num_hidden_layers
        keep = {f"encoder.layer.{i}." for i in range(n_layers - unfreeze, n_layers)}
        for name, prm in model.named_parameters():
            trainable = name.startswith("classifier") or any(k_ in name for k_ in keep)
            prm.requires_grad = trainable

        # class imbalance is handled in the loss, not by resampling
        w = torch.tensor(
            [len(y[tr]) / (2 * (y[tr] == 0).sum()), len(y[tr]) / (2 * (y[tr] == 1).sum())],
            dtype=torch.float,
        ).to(device)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        dl = DataLoader(
            DS([seqs[i] for i in tr], y[tr]),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        model.train()
        for ep in range(epochs):
            for bi, batch in enumerate(dl, 1):
                batch = {k_: v.to(device) for k_, v in batch.items()}
                labels = batch.pop("labels")
                loss = lossf(model(**batch).logits, labels)
                loss.backward()
                opt.step()
                opt.zero_grad()
                print(
                    f"  fold {k}/{folds} epoch {ep + 1}/{epochs} "
                    f"step {bi}/{len(dl)} loss {loss.item():.3f}",
                    end="\r",
                    flush=True,
                )

        model.eval()
        te_dl = DataLoader(
            DS([seqs[i] for i in te], y[te]),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        probs = []
        with torch.no_grad():
            for batch in te_dl:
                batch = {k_: v.to(device) for k_, v in batch.items()}
                batch.pop("labels")
                probs.append(
                    torch.softmax(model(**batch).logits, -1)[:, 1].cpu().numpy()
                )
        oof[te] = np.concatenate(probs)
        print(f"  fold {k}/{folds} done" + " " * 40)
        del model
    return oof


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(y, preds, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{name} (AUPRC {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax[1].axhline(np.mean(y), color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "roc_pr.png", dpi=160)

    fig, a = plt.subplots(figsize=(5, 4.6))
    for name, p in preds.items():
        pt, pp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        a.plot(pp, pt, "o-", label=f"{name} (ECE {expected_calibration_error(np.asarray(y), np.asarray(p)):.3f})")
    a.plot([0, 1], [0, 1], "k--", lw=0.8)
    a.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraction positive",
        title="Calibration",
    )
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "calibration.png", dpi=160)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", default="proteins_ad_pd_concordant.txt")
    ap.add_argument("--background", default=None,
                    help="file of gene symbols to draw negatives from "
                         "(strongly preferred: proteins quantified in ROSMAP "
                         "but not concordant). Length-matched at --ratio "
                         "unless --use-all-background is set")
    ap.add_argument("--use-all-background", action="store_true",
                    help="use every matched protein in --background as a "
                         "negative instead of a length-matched subsample")
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=".",
                    help="folder for results.json, predictions.csv, figures/")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.model = "facebook/esm2_t6_8M_UR50D"
        args.epochs, args.folds, args.max_len = 2, 3, 320

    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)

    proteome = download_proteome(HERE / "uniprot_human_reviewed.tsv")
    index = build_symbol_index(proteome)

    symbols = [
        s.strip()
        for s in Path(args.positives).read_text().split()
        if s.strip() and s.strip().lower() != "protein"
    ]
    pos, unmatched = resolve(symbols, proteome, index)
    (outdir / "unmatched_genes.txt").write_text("\n".join(unmatched) + "\n")
    print(
        f"\nPositives: {len(symbols)} symbols submitted, {len(pos)} unique UniProt "
        f"entries matched, {len(unmatched)} unmatched"
    )
    if unmatched:
        print("  unmatched:", ", ".join(unmatched))

    if args.background:
        bg_syms = [s.strip() for s in Path(args.background).read_text().split() if s.strip()]
        neg, bg_unmatched = resolve(bg_syms, proteome, index)
        neg = neg[~neg["accession"].isin(pos["accession"])]
        n_pool = len(neg)
        name = Path(args.background).name
        print(f"Background pool: {len(bg_syms)} symbols supplied, {n_pool} matched "
              f"and distinct from the positives ({len(bg_unmatched)} unmatched)")
        if args.use_all_background:
            bg_source = f"{name}, all {n_pool} proteins"
            print(f"Negatives: all {n_pool}, prevalence "
                  f"{len(pos) / (len(pos) + n_pool):.3f}")
            if not args.skip_finetune:
                print("  WARNING: fine-tuning on this many proteins takes hours. "
                      "Consider --skip-finetune for the full-background run.")
        else:
            neg = length_matched_background(pos, neg, args.ratio, rng_py)
            bg_source = (f"{name}, length-matched {args.ratio}:1 "
                         f"subsample of {n_pool}")
            print(f"Negatives: {len(neg)} length-matched from the pool of {n_pool}")
    else:
        pool = proteome[~proteome["accession"].isin(pos["accession"])]
        neg = length_matched_background(pos, pool, args.ratio, rng_py)
        bg_source = f"length-matched random reviewed human proteome (ratio {args.ratio}:1)"
        print(f"Negatives: {len(neg)} length-matched proteins sampled from the "
              f"reviewed human proteome")
        print("  NOTE: a random-proteome background lets the model separate on "
              "general brain/neuronal sequence character, not disease "
              "dysregulation specifically. Supply --background with the ROSMAP "
              "proteins that were quantified but not concordant for the "
              "interpretable version of this comparison.")

    data = pd.concat(
        [pos.assign(label=1), neg.assign(label=0)], ignore_index=True
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    seqs = data["sequence"].tolist()
    y = data["label"].to_numpy()
    print(f"\nDataset: {len(y)} proteins, {y.sum()} positive, prevalence "
          f"{y.mean():.3f}")

    device = pick_device()
    print(f"Model: {args.model} on {device}\n")

    preds, results = {}, {}

    print("Extracting frozen embeddings")
    X = embed(seqs, args.model, device, args.max_len, args.batch_size)
    preds["frozen"] = run_frozen(X, y, args.folds, args.seed)

    if not args.skip_finetune:
        print("\nFine-tuning ESM-2")
        preds["fine-tuned"] = run_finetune(
            seqs, y, args.model, device, args.max_len, args.batch_size,
            args.epochs, args.folds, args.unfreeze, args.lr, args.seed,
        )

    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 74)
    print(f"{'model':<28}{'AUROC':>9}{'AUPRC':>9}{'Brier':>9}{'ECE':>9}")
    print("-" * 74)
    for name, p in preds.items():
        m = score(y, p)
        m["auroc_ci95"] = bootstrap_ci(y, p, roc_auc_score, args.bootstrap, rng_np)
        results[name] = m
        ci = m["auroc_ci95"]
        print(
            f"{name:<28}{m['auroc']:>9.3f}{m['auprc']:>9.3f}"
            f"{m['brier']:>9.3f}{m['ece']:>9.3f}"
            + (f"   AUROC 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        )
    print("=" * 74)

    # permutation control, scored honestly against its own shuffled labels
    y_perm = rng_np.permutation(y)
    perm = run_frozen(X, y_perm, args.folds, args.seed)
    results["frozen (label permutation control)"] = score(y_perm, perm)
    print(
        "label-permutation control (frozen): AUROC "
        f"{results['frozen (label permutation control)']['auroc']:.3f} "
        "(expected near 0.500)"
    )

    make_figures(y, preds, outdir / "figures")

    pd.DataFrame(
        {
            "accession": data["accession"],
            "gene": data["gene_primary"],
            "length": data["length"],
            "label": y,
            **preds,
        }
    ).to_csv(outdir / "predictions.csv", index=False)

    (outdir / "results.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "device": str(device),
                "background_source": bg_source,
                "n_total": int(len(y)),
                "n_positive": int(y.sum()),
                "n_unmatched_symbols": len(unmatched),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote results.json, predictions.csv, figures/, unmatched_genes.txt to {outdir}/")


if __name__ == "__main__":
    sys.exit(main())
