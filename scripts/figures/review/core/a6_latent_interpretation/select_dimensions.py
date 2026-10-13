#!/usr/bin/env python
"""Pick the latent dimensions to show: decoded map vs the Neurosynth map of the dimension's top text-side term.

Reads outputs/associations.npz (run_analysis.py); writes outputs/neurosynth_agreement.csv (all 384 dimensions),
outputs/selected_dimensions_neurosynth.csv (the six shown) and outputs/cache/decoder_effects_all_<model>.npz, with the
model run_analysis.py used (outputs/results.json). CPU, ~25 s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _a6_common as C  # noqa: E402
import _final_model as M  # noqa: E402

# Cognitive Atlas terms that are generic words rather than a specific mental function
GENERIC = {"action", "activation level", "adaptation", "association", "balance", "competition", "concept", "detection",
           "discrimination", "efficiency", "excitation", "fixation", "focus", "generalization", "induction", "insight",
           "integration", "knowledge", "localization", "loss", "maintenance", "manipulation", "meaning", "morphology",
           "perception", "processing stage", "recognition", "resource", "routine", "rule", "schema", "search", "skill",
           "strategy", "strength", "thought", "transition", "utility", "demand", "load", "level"}
# modifiers, not concepts, when a multi-word term is shortened to a Neurosynth term
MODIFIERS = {"attentional", "behavioral", "biological", "central", "cognitive", "control", "delay", "disorder", "divergent",
             "error", "explicit", "feedback", "focused", "implicit", "interoceptive", "mental", "negative", "performance",
             "physical", "problem", "processing", "repetition", "resources", "response", "sequence", "source", "speed",
             "stage", "task", "theory", "time", "trait", "information"}
# shown: r >= MIN_R, peak voxel in the top 10 % of the Neurosynth map, |r| < MAX_OVERLAP_R with dims already chosen
MIN_R, MIN_PEAK_PCT, MAX_OVERLAP_R, N_SHOW, K_SIGMA = 0.3, 90.0, 0.4, 6, 2.0
# a term labels a dimension only if that dimension is among the term's 5 % most associated dimensions; without this,
# frequent terms (working memory, memory, retrieval) correlate weakly with every dimension and top many of them
TERM_TOP_FRAC = 0.05


def specific(A: np.ndarray, frac: float) -> np.ndarray:
    """[dims, terms] True where |r| of the dimension is in the top `frac` of that term's |r| over all dimensions."""
    if frac <= 0:
        return np.ones(A.shape, bool)
    k = max(1, int(round(frac * A.shape[0])))
    return np.abs(A) >= -np.sort(-np.abs(A), axis=0)[k - 1][None, :]


def neurosynth_term(term: str, index: dict) -> str | None:
    """Exact Neurosynth term, plural/singular, 'x of y' -> 'x y', else the longest specific sub-phrase; None if none."""
    for c in (term, term + "s", term[:-1] if term.endswith("s") else term, term.replace(" of ", " ")):
        if c in index:
            return c
    w = term.split()
    for n in range(len(w) - 1, 0, -1):
        for i in range(len(w) - n, -1, -1):  # later (head) phrases first
            c = " ".join(w[i:i + n])
            if c in index and c not in GENERIC and c not in MODIFIERS:
                return c
    return None


def decoder_effects(cache: Path) -> np.ndarray:
    """[dec(zbar + 2 sd e_j) - dec(zbar - 2 sd e_j)] / 2 for every dimension j (as in run_analysis.py), cached per model."""
    f = cache / f"decoder_effects_all_{M.model_tag()}.npz"
    if f.exists():
        return np.load(f)["delta"].astype(np.float32)
    ae = M.component("autoencoder", "cpu")
    lat, pm = C.load_pubmed_latents()  # stored latents == encoder output (results.json)
    Z = lat[C.split_masks(pm)["train"]].numpy()
    zbar, sd = Z.mean(0), Z.std(0)
    E = np.eye(Z.shape[1], dtype=np.float32) * (K_SIGMA * sd)[None, :]
    delta = 0.5 * (C.decode_latents(ae, zbar[None] + E, "cpu") - C.decode_latents(ae, zbar[None] - E, "cpu"))
    np.savez_compressed(f, delta=delta.astype(np.float32), zbar=zbar, sd=sd)
    return delta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    p.add_argument("--term-top-frac", type=float, default=TERM_TOP_FRAC,
                   help="Term specificity rule (0 = off: rank all terms by r, the earlier version).")
    args = p.parse_args()
    out = args.out
    C.set_offline_env()
    M.set_model_dir(M.REPO / C.read_json(out / "results.json")["config"]["model_dir"])  # the run run_analysis.py used
    cache = out / "cache"
    ns = C.Neurosynth(C.default_neurosynth_dir(), cache)
    a = np.load(out / "associations.npz", allow_pickle=True)
    terms, A_txt = [str(t) for t in a["terms"]], a["A_txt"]
    ok = np.array([t not in GENERIC for t in terms])[None, :] & specific(A_txt, args.term_top_frac)
    delta = decoder_effects(cache)
    dif = C.difumo(cache)
    rows = []
    for j in range(len(delta)):
        # top term by |r| with the text latent (specific terms only); its sign orients the dimension
        r_t = np.where(ok[j], A_txt[j], 0.0)
        t0 = int(np.argmax(np.abs(r_t)))
        s = 1 if r_t[t0] > 0 else -1
        top3 = [terms[i] for i in np.argsort(-s * r_t)[:3] if s * r_t[i] > 0]
        nst = neurosynth_term(terms[t0], ns.index)
        row = {"dim": j, "orientation": s, "top_term": terms[t0], "term_r": float(r_t[t0]), "top3_terms": "; ".join(top3),
               "neurosynth_term": nst or ""}
        if nst is not None:
            # r of the positive pole with the term's map; the opposite pole is shown if it fits better ("flipped")
            ref = ns.get(nst)
            same, opp = np.clip(s * delta[j], 0, None), np.clip(-s * delta[j], 0, None)
            r_same, r_opp = C.pearson(same, ref), C.pearson(opp, ref)
            flipped = r_opp > r_same
            shown = opp if flipped else same
            row |= {"r_matching_pole": r_same, "r_opposite_pole": r_opp, "flipped": bool(flipped),
                    "agreement_r": max(r_same, r_opp), "shown_sign": -s if flipped else s,
                    "neurosynth_pct_at_peak": 100.0 * float(np.mean(ref < ref[int(np.argmax(shown))])),
                    "top_difumo_shown": dif["names"][int(np.argmax(C.pearson_matrix(shown, dif["components"])))]}
        rows.append(row)
    df = pd.DataFrame(rows)
    cand = df[df["neurosynth_term"] != ""].sort_values("agreement_r", ascending=False)
    chosen, why = [], {}
    for r in cand.itertuples():
        if r.agreement_r < MIN_R or len(chosen) >= N_SHOW:
            break
        if r.neurosynth_pct_at_peak < MIN_PEAK_PCT:
            why[r.dim] = f"peak outside the top 10 % of the Neurosynth map (percentile {r.neurosynth_pct_at_peak:.0f})"
            continue
        m = np.clip(r.shown_sign * delta[r.dim], 0, None)
        dup = [c for c in chosen if r.neurosynth_term == df.loc[c, "neurosynth_term"]
               or abs(C.pearson(m, np.clip(df.loc[c, "shown_sign"] * delta[c], 0, None))) >= MAX_OVERLAP_R]
        if dup:
            why[r.dim] = f"same term as / near-duplicate map of dimension {dup[0]}"
            continue
        chosen.append(int(r.dim))
    df["skipped_because"] = df["dim"].map(why).fillna("")
    df.to_csv(out / "neurosynth_agreement.csv", index=False, float_format="%.4f")
    sel = df.loc[chosen]
    sel.to_csv(out / "selected_dimensions_neurosynth.csv", index=False, float_format="%.4f")
    v = df.dropna(subset=["agreement_r"])
    print(f"dimensions with a Neurosynth map: {len(v)} / {len(df)}; r >= {MIN_R}: {(v['agreement_r'] >= MIN_R).sum()}; "
          f"opposite pole matches better (flipped): {int(v['flipped'].sum())}")
    print(df[df["skipped_because"] != ""][["dim", "neurosynth_term", "agreement_r", "skipped_because"]].to_string(index=False))
    print(sel[["dim", "top3_terms", "neurosynth_term", "agreement_r", "neurosynth_pct_at_peak", "flipped", "top_difumo_shown"]].to_string(index=False))


if __name__ == "__main__":
    main()
