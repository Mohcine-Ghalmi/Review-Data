#!/usr/bin/env python3
"""AI-guided extraction for geostatistics applied to mine closure.

Uses the manually included mine-closure articles as positive training signal,
filters six raw source files, and exports results in the same 15-column format
as the raw source Excel files.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# NOTE: Users must specify their own data paths below.
REFERENCE_FILE = Path("")  # e.g., Path("my_reference_data.xlsx")

BASE_PATH = Path("")  # e.g., Path("my_data_directory")

OUTPUT_FILE = Path("")  # e.g., Path("filtered_output.xlsx")

# Update these to match your actual data structure.
SOURCE_FILES = {
    "IEEE Xplore": "",  # e.g., "IEEE Xplore/data.xlsx"
    "ScienceDirect": "",  # e.g., "ScienceDirect/data.xlsx"
    "scopus": "",  # e.g., "scopus/data.xlsx"
    "SpringerLink": "",  # e.g., "SpringerLink/data.xlsx"
    "Taylor & Francis": "",  # e.g., "Taylor & Francis/data.xlsx"
    "Web of Science": "",  # e.g., "Web of Science/data.xlsx"
}


GEOSTAT_TERMS = [
    r"\bgeostat\w*\b",
    r"\bkrig\w*\b",
    r"\bvariogram\w*\b",
    r"\bsemivariogram\w*\b",
    r"\bspatial interpolation\b",
    r"\bconditional simulation\b",
    r"\bstochastic simulation\b",
    r"\bsequential gaussian simulation\b",
    r"\bsequential indicator simulation\b",
    r"\bcokrig\w*\b",
    r"\bindicator kriging\b",
    r"\bordinary kriging\b",
]

MINE_CLOSURE_TERMS = [
    r"\bmine closure\b",
    r"\bclosure\b.*\bmine\b",
    r"\bmine rehabilitation\b",
    r"\bmine reclamation\b",
    r"\bpost.?mining\b",
    r"\bpost.?closure\b",
    r"\btailings\b",
    r"\btailings storage facilit\w*\b",
    r"\btailings dam\w*\b",
    r"\bTSF\b",
    r"\bmine waste\b",
    r"\bwaste rock\b",
    r"\boverburden\b",
    r"\bpit lake\b",
    r"\bacid mine drainage\b",
    r"\bAMD\b",
    r"\bseepage\b",
    r"\bdecommission\w*\b",
    r"\bsite restoration\b.*\bmine\b",
]

HARD_EXCLUDE = [
    r"\bsocial media\b|\btwitter\b|\bfacebook\b|\binstagram\b|\btiktok\b|\byoutube\b",
    r"\bhealth\b|\bmedical\b|\bmedicine\b|\bclinical\b|\bhospital\b|\bpatient\b|\bcancer\b|\bcovid\b",
    r"\btourism\b|\bstock market\b|\bmarketing\b|\bsentiment\b",
    r"\boil field\b|\bpetroleum\b|\bgas reservoir\b|\bwell log\b",
    r"\bagricultur\w*\b|\bcrop\b|\bfarming\b",
    r"\bdrug\b|\bpharmaceutical\b|\bbiomedical\b",
]

FINAL_EXCLUDE = [
    r"\bcorrection to\b|\berratum\b|\bcorrigendum\b",
    r"\becotoxic\w*\b|\bgeohealth\b|\bdiatoms?\b|\bdaphnia\b",
    r"\bgroundwater contamination\b",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def contains_any(text: pd.Series, patterns: List[str]) -> pd.Series:
    joined = "|".join(f"(?:{p})" for p in patterns)
    return text.str.contains(joined, regex=True, na=False)


def raw_text(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("display_name", "").map(normalize_text)
        + " "
        + df.get("keywords", "").map(normalize_text)
        + " "
        + df.get("abstract", "").map(normalize_text)
    ).str.lower()


def ref_text(df: pd.DataFrame) -> pd.Series:
    cols = {c.strip().lower(): c for c in df.columns}

    title_col = cols.get("titles") or cols.get("display_name")
    keywords_col = cols.get("keywords")
    journal_col = cols.get("journals") or cols.get("source")
    abstract_col = cols.get("abstract")

    return (
        df.get(title_col, "").map(normalize_text)
        + " "
        + df.get(keywords_col, "").map(normalize_text)
        + " "
        + df.get(journal_col, "").map(normalize_text)
        + " "
        + df.get(abstract_col, "").map(normalize_text)
    ).str.lower()


def load_sources() -> pd.DataFrame:
    frames = []
    for source, rel_path in SOURCE_FILES.items():
        path = BASE_PATH / rel_path
        if not path.exists():
            print(f"Missing source file: {path}")
            continue
        df = pd.read_excel(path)
        df["_source_folder"] = source
        df["source_file"] = str(path)
        frames.append(df)
        print(f"Loaded {source}: {len(df)}")
    if not frames:
        raise SystemExit("No source files loaded.")
    return pd.concat(frames, ignore_index=True, sort=False)


def dedupe_count(df_in: pd.DataFrame) -> int:
    if df_in.empty:
        return 0
    tmp = df_in.copy()
    tmp["_doi_norm"] = tmp.get("doi", "").map(normalize_text).str.strip().str.lower()
    tmp["_title_norm"] = tmp.get("display_name", "").map(normalize_text).str.strip().str.lower()
    with_doi = tmp[tmp["_doi_norm"] != ""].drop_duplicates(subset=["_doi_norm"], keep="first")
    without_doi = tmp[tmp["_doi_norm"] == ""].drop_duplicates(subset=["_title_norm"], keep="first")
    return len(with_doi) + len(without_doi)


def main() -> None:
    if not REFERENCE_FILE.exists():
        raise SystemExit(f"Reference file not found: {REFERENCE_FILE}")

    df_ref = pd.read_excel(REFERENCE_FILE)
    print(f"Reference loaded: {len(df_ref)} rows")

    df_all = load_sources()
    print(f"All sources loaded: {len(df_all)} rows")

    text_all = raw_text(df_all)
    in_geostat = contains_any(text_all, GEOSTAT_TERMS)
    in_closure = contains_any(text_all, MINE_CLOSURE_TERMS)
    out_offtopic = contains_any(text_all, HARD_EXCLUDE)

    mask_gate = (in_geostat | in_closure) & (~out_offtopic)
    df_gate = df_all[mask_gate].copy()
    print(f"After domain gate: {len(df_gate)} rows")

    pos_text = ref_text(df_ref)
    neg_pool = df_all[out_offtopic].copy()

    if len(neg_pool) == 0:
        neg_pool = df_all[~mask_gate].copy()

    neg_n = min(max(len(df_ref), 1500), len(neg_pool))
    neg_sample = neg_pool.sample(n=neg_n, random_state=42)
    neg_text = raw_text(neg_sample)

    X_train = pd.concat([pos_text, neg_text], ignore_index=True)
    y_train = [1] * len(pos_text) + [0] * len(neg_text)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=120000,
                    strip_accents="unicode",
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    gate_text = raw_text(df_gate)
    df_gate["_ai_score"] = model.predict_proba(gate_text)[:, 1]

    target_min_final = len(df_ref) + 1
    candidate_thresholds = [0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70, 0.68, 0.65, 0.62, 0.60]
    selected_threshold = candidate_thresholds[-1]
    best_threshold = candidate_thresholds[-1]
    best_count = 0

    for t in candidate_thresholds:
        c = dedupe_count(df_gate[df_gate["_ai_score"] >= t])
        if c > best_count:
            best_count = c
            best_threshold = t
        if c >= target_min_final:
            selected_threshold = t
            break
    else:
        selected_threshold = best_threshold

    df_keep = df_gate[df_gate["_ai_score"] >= selected_threshold].copy()
    print(f"After AI filter (threshold={selected_threshold:.2f}): {len(df_keep)} rows")

    final_text = raw_text(df_keep)
    noisy_mask = contains_any(final_text, FINAL_EXCLUDE)
    long_title_mask = df_keep.get("display_name", "").map(normalize_text).str.strip().str.len().ge(8)
    df_keep = df_keep[(~noisy_mask) & long_title_mask].copy()
    print(f"After final cleanup rules: {len(df_keep)} rows")

    df_keep["_cite_num"] = pd.to_numeric(df_keep.get("cited_by_count", 0), errors="coerce").fillna(0)
    df_keep = df_keep.sort_values(["_ai_score", "_cite_num"], ascending=[False, False])
    df_keep["_doi_norm"] = df_keep.get("doi", "").map(normalize_text).str.strip().str.lower()
    df_keep["_title_norm"] = df_keep.get("display_name", "").map(normalize_text).str.strip().str.lower()

    with_doi = df_keep[df_keep["_doi_norm"] != ""].drop_duplicates(subset=["_doi_norm"], keep="first")
    without_doi = df_keep[df_keep["_doi_norm"] == ""].drop_duplicates(subset=["_title_norm"], keep="first")
    df_final = pd.concat([with_doi, without_doi], ignore_index=True, sort=False)
    df_final = df_final.sort_values(["_ai_score", "_cite_num"], ascending=[False, False])
    df_final = df_final.drop_duplicates(subset=["_title_norm"], keep="first")

    if "publication_year" in df_final.columns:
        df_final = df_final.sort_values("publication_year", ascending=False)

    raw_cols = [
        "id",
        "link",
        "display_name",
        "doi",
        "doi_link",
        "publication_date",
        "publication_year",
        "type",
        "cited_by_count",
        "source",
        "abstract",
        "keywords",
        "authors",
        "affiliations",
        "countries",
    ]

    for col in raw_cols:
        if col not in df_final.columns:
            df_final[col] = ""

    out = df_final[raw_cols].copy().reset_index(drop=True)
    out.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved clean output: {OUTPUT_FILE} ({len(out)} rows)")


if __name__ == "__main__":
    main()
