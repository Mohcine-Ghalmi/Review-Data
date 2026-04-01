#!/usr/bin/env python3
"""AI-style filtering for mineral-resources geostatistics articles.

Pipeline:
1) Load curated reference set (positive examples).
2) Load all 6 raw source files.
3) Apply strict domain gates (must be geostatistics + mineral/mining context).
4) Train a lightweight text classifier using the curated set vs auto-mined negatives.
5) Keep only high-confidence in-domain records and save with same 15-column format.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import re

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
    "Web Of science": "",  # e.g., "Web of Science/data.xlsx"
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
    r"\bblock kriging\b",
    r"\bordinary kriging\b",
]

MINERAL_TERMS = [
    r"\bmineral\b",
    r"\bore\b",
    r"\bdeposit\b",
    r"\bresource estimation\b",
    r"\bgrade estimation\b",
    r"\bgrade control\b",
    r"\bdrillhole\b",
    r"\bdrill hole\b",
    r"\bmine\b",
    r"\bmining\b",
    r"\bopen pit\b",
    r"\bunderground mine\b",
    r"\bblock model\b",
]

HARD_EXCLUDE = [
    r"\bsocial media\b|\btwitter\b|\bfacebook\b|\binstagram\b|\btiktok\b|\byoutube\b",
    r"\bhealth\b|\bmedical\b|\bmedicine\b|\bclinical\b|\bhospital\b|\bpatient\b|\bcancer\b|\bcovid\b",
    r"\boil field\b|\bpetroleum\b|\bgas reservoir\b|\bwell log\b",
    r"\bagricultur\w*\b|\bcrop\b|\bfarming\b",
    r"\btourism\b|\bstock market\b|\bmarketing\b|\bsentiment\b",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def row_text(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("display_name", "").map(normalize_text)
        + " "
        + df.get("keywords", "").map(normalize_text)
        + " "
        + df.get("abstract", "").map(normalize_text)
    ).str.lower()


def contains_any(text: pd.Series, patterns: List[str]) -> pd.Series:
    joined = "|".join(f"(?:{p})" for p in patterns)
    return text.str.contains(joined, regex=True, na=False)


def load_sources() -> pd.DataFrame:
    frames = []
    for source, filename in SOURCE_FILES.items():
        path = BASE_PATH / source / filename
        if not path.exists():
            print(f"Missing source file: {path}")
            continue
        df = pd.read_excel(path)
        df["_source_folder"] = source
        frames.append(df)
        print(f"Loaded {source}: {len(df)}")

    if not frames:
        raise SystemExit("No source files loaded.")

    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    if not REFERENCE_FILE.exists():
        raise SystemExit(f"Reference file not found: {REFERENCE_FILE}")

    df_ref = pd.read_excel(REFERENCE_FILE)
    ref_cols = df_ref.columns.tolist()
    print(f"Reference loaded: {len(df_ref)} rows")

    df_all = load_sources()
    print(f"All sources loaded: {len(df_all)} rows")

    text_all = row_text(df_all)

    # Domain gate: must discuss both geostatistics and mineral/mining context.
    in_geostat = contains_any(text_all, GEOSTAT_TERMS)
    in_mineral = contains_any(text_all, MINERAL_TERMS)
    out_offtopic = contains_any(text_all, HARD_EXCLUDE)
    mask_gate = in_geostat & in_mineral & (~out_offtopic)

    df_gate = df_all[mask_gate].copy()
    print(f"After domain gate: {len(df_gate)} rows")

    # Build AI classifier (text semantics via TF-IDF + logistic regression).
    ref_text = row_text(df_ref)
    neg_pool = df_all[out_offtopic].copy()
    neg_pool_text = row_text(neg_pool)

    neg_sample_size = min(max(len(df_ref), 2000), len(neg_pool))
    if neg_sample_size == 0:
        neg_sample = df_all[~mask_gate].sample(n=min(len(df_ref), len(df_all[~mask_gate])), random_state=42)
        neg_sample_text = row_text(neg_sample)
    else:
        neg_sample = neg_pool.sample(n=neg_sample_size, random_state=42)
        neg_sample_text = row_text(neg_sample)

    X_train = pd.concat([ref_text, neg_sample_text], ignore_index=True)
    y_train = [1] * len(ref_text) + [0] * len(neg_sample_text)

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
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)),
        ]
    )
    model.fit(X_train, y_train)

    gate_text = row_text(df_gate)
    proba = model.predict_proba(gate_text)[:, 1]
    df_gate["_ai_score"] = proba

    def dedupe_count(df_in: pd.DataFrame) -> int:
        if df_in.empty:
            return 0
        tmp = df_in.copy()
        tmp["_doi_norm"] = tmp.get("doi", "").map(normalize_text).str.strip().str.lower()
        tmp["_title_norm"] = tmp.get("display_name", "").map(normalize_text).str.strip().str.lower()
        with_doi_tmp = tmp[tmp["_doi_norm"] != ""].drop_duplicates(subset=["_doi_norm"], keep="first")
        without_doi_tmp = tmp[tmp["_doi_norm"] == ""].drop_duplicates(subset=["_title_norm"], keep="first")
        return len(with_doi_tmp) + len(without_doi_tmp)

    target_min_final = len(df_ref) + 1
    candidate_thresholds = [0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70, 0.68, 0.65, 0.62, 0.60]

    selected_threshold = candidate_thresholds[-1]
    selected_count = 0
    for t in candidate_thresholds:
        count_t = dedupe_count(df_gate[df_gate["_ai_score"] >= t])
        if count_t >= target_min_final:
            selected_threshold = t
            selected_count = count_t
            break

    df_keep = df_gate[df_gate["_ai_score"] >= selected_threshold].copy()
    print(f"After AI filter (threshold={selected_threshold:.2f}): {len(df_keep)} rows")
    if selected_count:
        print(f"Estimated final deduplicated rows at selected threshold: {selected_count}")

    df_keep = df_keep.sort_values("_ai_score", ascending=False)
    df_keep["_doi_norm"] = df_keep.get("doi", "").map(normalize_text).str.strip().str.lower()
    df_keep["_title_norm"] = df_keep.get("display_name", "").map(normalize_text).str.strip().str.lower()

    with_doi = df_keep[df_keep["_doi_norm"] != ""].drop_duplicates(subset=["_doi_norm"], keep="first")
    without_doi = df_keep[df_keep["_doi_norm"] == ""].drop_duplicates(subset=["_title_norm"], keep="first")
    df_final = pd.concat([with_doi, without_doi], ignore_index=True, sort=False)

    if "publication_year" in df_final.columns:
        df_final = df_final.sort_values("publication_year", ascending=False)

    for col in ref_cols:
        if col not in df_final.columns:
            df_final[col] = ""
    df_final = df_final[ref_cols].reset_index(drop=True)

    df_final.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved clean output: {OUTPUT_FILE} ({len(df_final)} rows)")


if __name__ == "__main__":
    main()
