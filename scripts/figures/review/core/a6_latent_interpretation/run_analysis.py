#!/usr/bin/env python
"""What do the 384 autoencoder latent dimensions encode? Decoder effect maps and encoder associations vs DiFuMo,
text/image latent associations with Cognitive Atlas terms, agreement vs a dimension-shuffled null. Autoencoder, latents
and text-to-brain head of --model-dir (default runs/final: head v2).

Run: python run_analysis.py [--smoke] [--device cpu]   (outputs -> ./outputs)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _a6_common as C  # noqa: E402
import _final_model as M  # noqa: E402

log = C.get_logger("a6_latent")
# Cognitive Atlas entries that are generic study-design words in abstracts
GENERIC_TERMS = {"activation", "stimulus", "response", "task", "context", "network", "control", "goal", "image", "signal",
                 "brain", "mind", "rest", "set", "time", "error", "feedback", "target", "cue", "condition", "state"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    M.add_model_args(p)
    p.add_argument("--device", default="auto", help="Device (auto/cpu/cuda).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--smoke", action="store_true", help="Quick run: 2000 papers, 150 terms.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap on training papers.")
    p.add_argument("--max-terms", type=int, default=None, help="Cap on Cognitive Atlas terms.")
    p.add_argument("--min-df", type=int, default=30, help="Min. training papers mentioning a term.")
    p.add_argument("--k-sigma", type=float, default=2.0, help="Latent step (in sd) for the decoder effect.")
    p.add_argument("--n-select", type=int, default=6, help="Dimensions to select.")
    p.add_argument("--max-overlap-r", type=float, default=0.4, help="Max |r| between effect maps of selected dims.")
    p.add_argument("--n-null", type=int, default=200, help="Dimension shuffles for the null.")
    p.add_argument("--force", action="store_true", help="Ignored (always recomputes).")
    a = p.parse_args()
    if a.smoke:
        a.max_samples = a.max_samples or 2000
        a.max_terms = a.max_terms or 150
        a.min_df = min(a.min_df, 10)
        a.n_null = min(a.n_null, 50)
    return a


def colcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between every column of A [N, p] and of B [N, q] -> [p, q]."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / np.clip(np.linalg.norm(A, axis=0), 1e-12, None)
    B = B / np.clip(np.linalg.norm(B, axis=0), 1e-12, None)
    return A.T @ B


def term_presence(texts: pd.Series, terms: list[str]) -> np.ndarray:
    low = texts.fillna("").str.lower()
    out = np.zeros((len(low), len(terms)), np.float32)
    for j, t in enumerate(terms):
        out[:, j] = low.str.contains(r"\b" + re.escape(t) + r"s?\b", regex=True).to_numpy()
    return out


def topk_overlap(A: np.ndarray, B: np.ndarray, k: int = 5) -> np.ndarray:
    ta = np.argsort(-A, axis=1)[:, :k]
    tb = np.argsort(-B, axis=1)[:, :k]
    return np.array([len(set(a) & set(b)) / k for a, b in zip(ta, tb)])


def rowcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.mean(C.zrows(A) * C.zrows(B), axis=1)


def null_stats(obs: np.ndarray, null: list) -> dict:
    """Observed mean, null mean and permutation p (one-sided, +1 corrected)."""
    return {"mean": float(obs.mean()), "null_mean": float(np.mean(null)),
            "null_p": float((np.sum(np.asarray(null) >= obs.mean()) + 1) / (len(null) + 1))}


def main() -> None:
    args = parse_args()
    C.set_offline_env()
    C.seed_all(args.seed)
    M.from_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "cache"
    device = C.select_device(args.device)
    timings = C.Timings(args.out / "timings.json")
    rng = np.random.default_rng(args.seed)
    import torch

    log.info("device=%s model=%s smoke=%s out=%s", device, M.model_tag(), args.smoke, args.out)
    with timings.time("load"):
        ae = M.component("autoencoder", device)
        head = M.component("proj_head_text_mse", device)
        dif = C.difumo(cache)
        D = dif["components"]  # [512, V]
        lat_stored, pm = C.load_pubmed_latents()
        tr = C.take_prefix(C.split_masks(pm)["train"], args.max_samples)
        pm_tr = pm[tr]
    with timings.time("encode_maps", n=int(tr.sum())):
        X = C.load_pubmed_maps(pm_tr)  # [N, V] training maps
        with torch.no_grad():
            Z = torch.cat([ae.encoder(X[s:s + 1024].to(device)).float().cpu() for s in range(0, len(X), 1024)]).numpy()
        stored_r = float(np.mean(C.zrows(Z[:200]) * C.zrows(lat_stored[tr][:200].numpy())))
        coef = (X.numpy() @ D.T) / np.clip(D.sum(1), 1e-8, None)[None, :]  # DiFuMo-weighted mean activation [N, 512]
        del X
    log.info("encoded %d maps (row corr. with stored latents %.3f)", len(Z), stored_r)

    with timings.time("decoder_effects"):
        zbar, sd = Z.mean(0), Z.std(0)
        n_dim = Z.shape[1]
        E = np.eye(n_dim, dtype=np.float32) * (args.k_sigma * sd)[None, :]
        plus = C.decode_latents(ae, zbar[None, :] + E, device)
        minus = C.decode_latents(ae, zbar[None, :] - E, device)
        delta = 0.5 * (plus - minus)  # [384, V]: [dec(zbar + k sd_j e_j) - dec(zbar - k sd_j e_j)] / 2
        A_dec = C.pearson_matrix(delta, D)  # [384, 512]
        # share of |delta_j| in its top 5 % voxels
        sparsity = np.array([np.sort(np.abs(d))[::-1][: int(0.05 * d.size)].sum() / max(np.abs(d).sum(), 1e-12) for d in delta])
        effect_size = np.abs(delta).max(1)
    with timings.time("encoder_difumo_assoc"):
        A_img = colcorr(Z, coef).astype(np.float32)  # [384, 512]

    with timings.time("text_side"):
        rows = C.load_pubmed_text_rows(pm_tr)
        texts = rows["name"] + " " + rows["description"]
        S = torch.nn.functional.normalize(C.load_specter_latents(pm_tr), dim=1)
        with torch.no_grad():
            Zt = torch.cat([head(S[s:s + 2048].to(device)).float().cpu() for s in range(0, len(S), 2048)]).numpy()
        cog_file = C.data_dir() / "cogatlas.parquet"
        if cog_file.exists():
            cog = pd.read_parquet(cog_file)
        else:
            from _hf_pins import load_dataset

            cog = load_dataset("cogatlas")
        terms_all = sorted({str(t).strip().lower() for t in cog["term"].dropna() if len(str(t).strip()) > 2} - GENERIC_TERMS)
        if args.max_terms:
            # frequent terms first, then an alphabetical prefix
            common = ["attention", "working memory", "reward", "pain", "fear", "language", "memory", "emotion", "perception",
                      "motor control", "decision making", "face perception", "reading", "inhibition", "anxiety"]
            terms_all = [t for t in common if t in terms_all] + [t for t in terms_all if t not in common][: args.max_terms]
        P = term_presence(texts, terms_all)
        df_count = P.sum(0)
        keep = df_count >= args.min_df
        terms = [t for t, k in zip(terms_all, keep) if k]
        P = P[:, keep]
        A_txt = colcorr(Zt, P).astype(np.float32)       # projected-text latent vs term presence [384, T]
        A_imgtxt = colcorr(Z, P).astype(np.float32)     # image latent vs term presence [384, T]
        A_txt, A_imgtxt = np.nan_to_num(A_txt), np.nan_to_num(A_imgtxt)
    log.info("terms kept: %d / %d (min_df=%d)", len(terms), len(terms_all), args.min_df)
    with timings.time("term_embeddings", n=len(terms)):
        sp = M.component("specter", device)
        Et = C.encode_specter(sp, terms)
        with torch.no_grad():
            Zterm = head(Et.to(device)).float().cpu().numpy()  # [T, 384]
        A_emb = ((Zterm - Zterm.mean(0)) / np.clip(Zterm.std(0), 1e-8, None)).T  # [384, T]
        del sp

    with timings.time("agreement_and_selection"):
        ov = topk_overlap(A_dec, A_img, 5)
        agree_terms = rowcorr(A_txt, A_imgtxt)
        agree_emb = rowcorr(A_txt, A_emb)
        null_ov, null_terms, null_emb = [], [], []
        for _ in range(args.n_null):
            perm = rng.permutation(n_dim)
            null_ov.append(topk_overlap(A_dec, A_img[perm], 5).mean())
            null_terms.append(rowcorr(A_txt, A_imgtxt[perm]).mean())
            null_emb.append(rowcorr(A_txt, A_emb[perm]).mean())
        agreement = {
            "difumo_top5_overlap_decoder_vs_encoder": null_stats(ov, null_ov) | {"chance": 5 / D.shape[0]},
            "term_assoc_text_vs_image_latent_r": null_stats(agree_terms, null_terms),
            "term_assoc_text_vs_term_embedding_r": null_stats(agree_emb, null_emb),
        }
        dec_sel = np.abs(A_dec).max(1)
        txt_sel = np.abs(A_txt).max(1)

        def pct_rank(x):
            return pd.Series(x).rank(pct=True).to_numpy()

        score = np.mean([pct_rank(dec_sel), pct_rank(sparsity), pct_rank(txt_sel), pct_rank(agree_terms), pct_rank(effect_size)], axis=0)
        # orient each dimension so that its decoder effect correlates positively with its best DiFuMo component
        sign = np.sign(A_dec[np.arange(n_dim), np.abs(A_dec).argmax(1)])
        sign[sign == 0] = 1
        order = np.argsort(-score)
        selected, used_comp, used_term = [], set(), set()
        for j in order:
            cj = int(np.argmax(sign[j] * A_dec[j]))
            tj = int(np.argmax(sign[j] * A_txt[j]))
            if cj in used_comp or tj in used_term:
                continue
            # spatial diversity: decoder effect maps of the selected dimensions correlate < --max-overlap-r
            if selected and np.max(np.abs(C.pearson_matrix(delta[j], delta[selected]))) > args.max_overlap_r:
                continue
            selected.append(int(j))
            used_comp.add(cj)
            used_term.add(tj)
            if len(selected) >= args.n_select:
                break

    names = dif["names"]
    rows_out = []
    for j in range(n_dim):
        s = sign[j]
        dj, ij, tj, it, ej = s * A_dec[j], s * A_img[j], s * A_txt[j], s * A_imgtxt[j], s * A_emb[j]
        r = {"dim": j, "sign": int(s), "score": float(score[j]), "rank": int(np.where(order == j)[0][0]) + 1,
             "selected": j in selected, "latent_std": float(sd[j]), "effect_max": float(effect_size[j]),
             "spatial_top5pct_mass": float(sparsity[j]), "decoder_difumo_max_abs_r": float(dec_sel[j]),
             "text_term_max_abs_r": float(txt_sel[j]), "difumo_top5_overlap_dec_enc": float(ov[j]),
             "term_agreement_text_image": float(agree_terms[j]), "term_agreement_text_embedding": float(agree_emb[j])}
        for lab, vec, lst, k in (("decoder_difumo", dj, names, 3), ("encoder_difumo", ij, names, 3),
                                 ("text_terms_pos", tj, terms, 5), ("image_terms_pos", it, terms, 5), ("embedding_terms_pos", ej, terms, 5)):
            top = np.argsort(-vec)[:k]
            r[lab] = "; ".join(f"{lst[i]} ({vec[i]:.2f})" for i in top)
        neg = np.argsort(tj)[:3]
        r["text_terms_neg"] = "; ".join(f"{terms[i]} ({tj[i]:.2f})" for i in neg)
        negc = np.argsort(dj)[:2]
        r["decoder_difumo_neg"] = "; ".join(f"{names[i]} ({dj[i]:.2f})" for i in negc)
        rows_out.append(r)
    summ = pd.DataFrame(rows_out)
    summ.to_csv(args.out / "latent_dimension_summary.csv", index=False)
    summ[summ["selected"]].sort_values("rank").to_csv(args.out / "selected_dimensions.csv", index=False)

    # top 10 DiFuMo components and terms per dimension
    long = []
    for j in range(n_dim):
        s = sign[j]
        for i in np.argsort(-(s * A_dec[j]))[:10]:
            long.append({"dim": j, "kind": "decoder_difumo", "label": names[i], "yeo7": dif["yeo7"][i], "r": float(s * A_dec[j, i])})
        for i in np.argsort(-(s * A_txt[j]))[:10]:
            long.append({"dim": j, "kind": "text_term", "label": terms[i], "yeo7": "", "r": float(s * A_txt[j, i])})
    pd.DataFrame(long).to_csv(args.out / "top_associations_long.csv", index=False)

    np.savez_compressed(args.out / "associations.npz", A_dec=A_dec.astype(np.float32), A_img=A_img, A_txt=A_txt, A_imgtxt=A_imgtxt,
                        A_emb=A_emb.astype(np.float32), terms=np.array(terms), difumo_names=names, sign=sign,
                        selected=np.array(selected), delta_selected=(sign[selected, None] * delta[selected]).astype(np.float32),
                        null_overlap=np.array(null_ov), null_terms=np.array(null_terms), overlap=ov, agree_terms=agree_terms,
                        term_df=df_count[keep])
    nii = args.out / "nifti"
    nii.mkdir(exist_ok=True)
    for stale in nii.glob("latent_dim*_decoder_effect.nii.gz"):  # selection may change between runs
        stale.unlink()
    for j in selected:
        C.unmask(sign[j] * delta[j]).to_filename(nii / f"latent_dim{j:03d}_decoder_effect.nii.gz")
    res = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"device": str(device)} | M.model_info(),
           "n_train_maps": int(len(Z)), "n_terms": len(terms), "n_terms_candidates": len(terms_all),
           "encoder_vs_stored_latent_rowcorr": stored_r, "agreement": agreement, "selected_dims": selected,
           "selected": summ[summ["selected"]].sort_values("rank")[["dim", "score", "decoder_difumo", "text_terms_pos"]].to_dict(orient="records"),
           "versions": C.installed_versions()}
    C.write_json(args.out / "results.json", res)
    log.info("agreement: %s", agreement)
    log.info("selected dims: %s", selected)
    log.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
