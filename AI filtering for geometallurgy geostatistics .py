#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
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
    "Web of science": "",  # e.g., "Web of Science/data.xlsx"
}

PLATFORM_CANONICAL = {
    "ieee xplore": "IEEE Xplore",
    "sciencedirect": "ScienceDirect",
    "scopus": "Scopus",
    "springerlink": "SpringerLink",
    "taylor & farncis": "Taylor & Francis",
    "taylor & francis": "Taylor & Francis",
    "web of science": "Web of Science",
}

VALID_PLATFORMS = set(PLATFORM_CANONICAL.values())

MIN_TOTAL_MATCHES = 2

GROUP_A_TERMS = [
    "geostatistic",
    "kriging",
    "semivariogram",
    "variogram",
    "conditional simulation",
    "sequential gaussian simulation",
    "sequential indicator simulation",
    "multiple-point geostatistics",
    "cokriging",
    "co-kriging",
    "indicator kriging",
    "block kriging",
    "ordinary kriging",
    "universal kriging",
    "variogram model",
    "spatial continuity",
    "nugget effect",
]

GROUP_B_TERMS = [
    "geometallurgy",
    "geometallurgical",
    "geo-metallurgy",
    "geo-metallurgical",
    "mineral processing",
    "extractive metallurgy",
    "mineral beneficiation",
    "ore processing",
    "flotation",
    "geometallurgical model",
    "geometallurgical characterization",
    "ore hardness",
    "ore sorting",
    "metallurgical recovery",
    "grade recovery",
    "resource to recovery",
    "recovery",
    "metallurgical performance",
    "reprocessing",
    "ore texture",
]
EXCLUDE_TERMS = [
    "petroleum",
    "oil and gas",
    "gas reservoir",
    "shale",
    "aquifer",
    "groundwater",
    "hydrogeolog",
    "agriculture",
    "soil contamination",
    "medical",
    "medecine",
    "contamination",
    "contaminant",
    "contaminated",
    "pollution",
    "microplastics",
    "reservoir",
]


GEOSTAT_TERMS = [
    r"\bgeostat\w*\b",
    r"\bkrig\w*\b",
    r"\bblock kriging\b",
    r"\bvariogram\w*\b",
    r"\bsemivariogram\w*\b",
    r"\bspatial interpolation\b",
    r"\bspatial continuity\b",
    r"\bconditional simulation\b",
    r"\bstochastic simulation\b",
    r"\bgeostatistical simulation\b",
    r"\bsequential gaussian simulation\b",
    r"\bsequential indicator simulation\b",
    r"\b\bsgs\b\b",
    r"\b\bsis\b\b",
    r"\bmultiple[ -]?point geostat\w*\b",
    r"\bmps\b",
    r"\bmpstats\b",
    r"\bcokrig\w*\b",
    r"\bindicator kriging\b",
    r"\bordinary kriging\b",
    r"\bresource model\w*\b",
    r"\bgrade model\w*\b",
]

GEOMET_TERMS = [
    r"\bgeometall\w*\b",
    r"\bgeological[- ]metallurgical\b",
    r"\bprocess mineralog\w*\b",
    r"\bmineral process\w*\b",
    r"\bore process\w*\b",
    r"\bextractive metallurg\w*\b",
    r"\bbeneficiation\b",
    r"\bmetallurgical\b.*\bresponse\b",
    r"\bmetallurgical\b.*\brecovery\b",
    r"\brecovery\b.*\bmetallurgical\b",
    r"\bgrade\b.*\brecovery\b",
    r"\bmetallurgical accounting\b",
    r"\bmine.?to.?mill\b",
    r"\bmine.?to.?metal\b",
    r"\bresource.?to.?recovery\b",
    r"\bpit.?to.?plant\b",
    r"\bflotation\b",
    r"\bcomminution\b",
    r"\bheap leach\w*\b",
    r"\bbioleach\w*\b",
    r"\bgrindability\b",
    r"\bore hardness\b",
    r"\bbond work index\b",
    r"\bSAG\b.*\bmill\b",
    r"\bHPGR\b",
    r"\bdrop weight test\b",
    r"\bQEMSCAN\b",
    r"\bautomated mineralogy\b",
    r"\bmineral liberation\b",
    r"\bMLA\b",
    r"\bore variability\b",
    r"\bdeportment\b",
    r"\belement\w* deportment\b",
    r"\bmineralogical domaining\b",
]

MINING_CONTEXT_TERMS = [
    r"\bmining\b",
    r"\bmine\b",
    r"\bore\b",
    r"\bdeposit\w*\b",
    r"\bmineral\w*\b",
    r"\bresource estimation\b",
    r"\breserve estimation\b",
    r"\bgrade control\b",
    r"\bblock model\w*\b",
    r"\bpit\b",
    r"\bplant\b",
]

HARD_EXCLUDE = [
    r"\bsocial media\b|\btwitter\b|\bfacebook\b|\binstagram\b|\btiktok\b|\byoutube\b",
    r"\bhealth\b|\bmedical\b|\bmedicine\b|\bclinical\b|\bhospital\b|\bpatient\b|\bcancer\b|\bcovid\b",
    r"\btourism\b|\bstock market\b|\bmarketing\b|\bsentiment\b",
    r"\boil field\b|\bpetroleum\b|\bgas reservoir\b|\bwell log\b",
    r"\bagricultur\w*\b|\bcrop\b|\bfarming\b",
    r"\bdrug\b|\bpharmaceutical\b|\bbiomedical\b",
    r"\bwind turbine\b|\bsolar energy\b",
]

FINAL_EXCLUDE = [
    r"\bcorrection to\b|\berratum\b|\bcorrigendum\b",
    r"\becotoxic\w*\b|\bgeohealth\b|\bdiatoms?\b|\bdaphnia\b",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def contains_any(text: pd.Series, patterns: List[str]) -> pd.Series:
    joined = "|".join(f"(?:{p})" for p in patterns)
    return text.str.contains(joined, regex=True, na=False)


def normalize_for_keyword_matching(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text).lower().strip()
    return re.sub(r"\s+", " ", value)


def count_term_hits(text: str, terms: List[str]) -> List[str]:
    normalized = normalize_for_keyword_matching(text)
    hits: List[str] = []
    for term in terms:
        if term in normalized:
            hits.append(term)
    return hits


def build_keyword_blob(row: pd.Series) -> str:
    title = row.get("display_name", "")
    abstract = row.get("abstract", "")
    keywords = row.get("keywords", "")
    return f"{title} {abstract} {keywords}".strip()


def is_relevant_by_groups(row: pd.Series) -> bool:
    blob = build_keyword_blob(row)
    a_hits = count_term_hits(blob, GROUP_A_TERMS)
    b_hits = count_term_hits(blob, GROUP_B_TERMS)
    x_hits = count_term_hits(blob, EXCLUDE_TERMS)

    total_hits = len(set(a_hits + b_hits))
    return bool(a_hits) and bool(b_hits) and total_hits >= MIN_TOTAL_MATCHES and not x_hits


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
    title_col = cols.get("titles", "")
    keywords_col = cols.get("keywords", "")
    journals_col = cols.get("journals", "")

    return (
        df.get(title_col, "").map(normalize_text)
        + " "
        + df.get(keywords_col, "").map(normalize_text)
        + " "
        + df.get(journals_col, "").map(normalize_text)
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
    tmp["_source_file_norm"] = tmp.get("source_file", "").map(normalize_text).str.strip().str.lower()
    with_doi = tmp[tmp["_doi_norm"] != ""].drop_duplicates(subset=["_doi_norm"], keep="first")
    without_doi = tmp[tmp["_doi_norm"] == ""].drop_duplicates(subset=["_title_norm", "_source_file_norm"], keep="first")
    return len(with_doi) + len(without_doi)


def build_reference_keys(df_ref: pd.DataFrame) -> tuple[set[str], set[str]]:
    cols = {c.strip().lower(): c for c in df_ref.columns}
    title_col = cols.get("titles")

    doi_col = None
    for key, original in cols.items():
        if key == "doi" or "doi" in key:
            doi_col = original
            break

    if title_col:
        ref_titles = (
            df_ref[title_col]
            .map(normalize_text)
            .str.strip()
            .str.lower()
            .loc[lambda s: s != ""]
        )
    else:
        ref_titles = pd.Series(dtype=str)

    if doi_col:
        ref_dois = (
            df_ref[doi_col]
            .map(normalize_text)
            .str.strip()
            .str.lower()
            .loc[lambda s: s != ""]
        )
    else:
        ref_dois = pd.Series(dtype=str)

    return set(ref_dois.tolist()), set(ref_titles.tolist())


def main() -> None:
    if not REFERENCE_FILE.exists():
        raise SystemExit(f"Reference file not found: {REFERENCE_FILE}")

    df_ref = pd.read_excel(REFERENCE_FILE)
    print(f"Reference loaded: {len(df_ref)} rows")
    ref_doi_set, ref_title_set = build_reference_keys(df_ref)

    df_all = load_sources()
    print(f"All sources loaded: {len(df_all)} rows")

    text_all = raw_text(df_all)
    in_geostat = contains_any(text_all, GEOSTAT_TERMS)
    in_geomet = contains_any(text_all, GEOMET_TERMS)
    in_mining = contains_any(text_all, MINING_CONTEXT_TERMS)
    out_offtopic = contains_any(text_all, HARD_EXCLUDE)

    strict_mask = in_geostat & in_geomet & in_mining
    print(f"Strict dual-domain matches before exclusions: {int(strict_mask.sum())}")

    mask_gate = strict_mask & (~out_offtopic)

    df_gate = df_all[mask_gate].copy()
    print(f"After domain gate: {len(df_gate)} rows")

    df_gate = df_gate[df_gate.apply(is_relevant_by_groups, axis=1)].copy()
    print(f"After Group A/B keyword rule: {len(df_gate)} rows")

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

    df_gate["_ai_score"] = model.predict_proba(raw_text(df_gate))[:, 1]

    ref_corpus = pos_text[pos_text.str.strip() != ""].reset_index(drop=True)
    cand_corpus = raw_text(df_gate).reset_index(drop=True)

    if len(ref_corpus) > 0 and len(cand_corpus) > 0:
        sim_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=80000,
            strip_accents="unicode",
        )
        sim_matrix = sim_vectorizer.fit_transform(pd.concat([ref_corpus, cand_corpus], ignore_index=True))
        ref_matrix = sim_matrix[: len(ref_corpus)]
        cand_matrix = sim_matrix[len(ref_corpus) :]
        ref_centroid = np.asarray(ref_matrix.mean(axis=0)).reshape(1, -1)
        df_gate["_ref_sim"] = cosine_similarity(cand_matrix, ref_centroid).ravel()
    else:
        df_gate["_ref_sim"] = 0.0

    df_gate["_final_score"] = (0.7 * df_gate["_ai_score"]) + (0.3 * df_gate["_ref_sim"])

    target_n = max(len(df_ref) + 20, 120)
    target_min_final = max(len(df_ref), 80)
    candidate_thresholds = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.80,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
        0.68,
        0.66,
        0.64,
        0.62,
        0.60,
        0.58,
        0.56,
        0.54,
        0.52,
        0.50,
        0.48,
        0.46,
        0.44,
        0.42,
        0.40,
    ]
    selected_threshold = candidate_thresholds[-1]
    best_threshold = candidate_thresholds[-1]
    best_gap = float("inf")

    for t in candidate_thresholds:
        c = dedupe_count(df_gate[df_gate["_final_score"] >= t])
        gap = abs(c - target_n)
        if gap < best_gap:
            best_gap = gap
            best_threshold = t
        if c >= target_min_final and c <= int(target_n * 1.50):
            selected_threshold = t
            break
    else:
        selected_threshold = best_threshold

    df_keep_base = df_gate[df_gate["_final_score"] >= selected_threshold].copy()

    relaxed_threshold = max(selected_threshold - 0.15, 0.35)
    df_source_boost = (
        df_gate[df_gate["_final_score"] >= relaxed_threshold]
        .sort_values(["_final_score", "_ai_score"], ascending=[False, False])
        .groupby("_source_folder", as_index=False, group_keys=False)
        .head(12)
    )

    df_keep = pd.concat([df_keep_base, df_source_boost], ignore_index=False)
    df_keep = df_keep.drop_duplicates(subset=["doi", "display_name", "source_file"], keep="first")

    print(
        f"After AI+reference filter (threshold={selected_threshold:.2f}, source boost={len(df_source_boost)}): {len(df_keep)} rows"
    )

    noisy_mask = contains_any(raw_text(df_keep), FINAL_EXCLUDE)
    long_title_mask = df_keep.get("display_name", "").map(normalize_text).str.strip().str.len().ge(8)
    strict_text = raw_text(df_keep)
    strict_domain_mask = (
        contains_any(strict_text, GEOSTAT_TERMS)
        & contains_any(strict_text, GEOMET_TERMS)
        & contains_any(strict_text, MINING_CONTEXT_TERMS)
    )
    df_keep = df_keep[(~noisy_mask) & long_title_mask & strict_domain_mask].copy()
    print(f"After final cleanup rules: {len(df_keep)} rows")

    df_keep["_cite_num"] = pd.to_numeric(df_keep.get("cited_by_count", 0), errors="coerce").fillna(0)
    df_keep = df_keep.sort_values(["_final_score", "_ai_score", "_cite_num"], ascending=[False, False, False])

    df_keep["_doi_norm"] = df_keep.get("doi", "").map(normalize_text).str.strip().str.lower()
    df_keep["_title_norm"] = df_keep.get("display_name", "").map(normalize_text).str.strip().str.lower()
    df_keep["_source_file_norm"] = df_keep.get("source_file", "").map(normalize_text).str.strip().str.lower()

    with_doi = df_keep[df_keep["_doi_norm"] != ""].drop_duplicates(subset=["_doi_norm"], keep="first")
    without_doi = df_keep[df_keep["_doi_norm"] == ""].drop_duplicates(subset=["_title_norm", "_source_file_norm"], keep="first")
    df_final = pd.concat([with_doi, without_doi], ignore_index=True, sort=False)
    df_final = df_final.sort_values(["_final_score", "_ai_score", "_cite_num"], ascending=[False, False, False])

    gate_noisy_mask = contains_any(raw_text(df_gate), FINAL_EXCLUDE)
    gate_long_title_mask = df_gate.get("display_name", "").map(normalize_text).str.strip().str.len().ge(8)
    df_gate_clean = df_gate[(~gate_noisy_mask) & gate_long_title_mask].copy()
    df_gate_clean["_cite_num"] = pd.to_numeric(df_gate_clean.get("cited_by_count", 0), errors="coerce").fillna(0)
    df_gate_clean["_doi_norm"] = df_gate_clean.get("doi", "").map(normalize_text).str.strip().str.lower()
    df_gate_clean["_title_norm"] = df_gate_clean.get("display_name", "").map(normalize_text).str.strip().str.lower()
    df_gate_clean = df_gate_clean.sort_values(["_final_score", "_ai_score", "_cite_num"], ascending=[False, False, False])

    ref_in_sources = df_gate_clean[
        df_gate_clean["_doi_norm"].isin(ref_doi_set) | df_gate_clean["_title_norm"].isin(ref_title_set)
    ].copy()
    if not ref_in_sources.empty:
        ref_in_sources = (
            ref_in_sources
            .sort_values(["_cite_num", "_final_score", "_ai_score"], ascending=[False, False, False])
            .groupby("_source_folder", as_index=False, group_keys=False)
            .head(20)
        )
        ref_with_doi = ref_in_sources[ref_in_sources["_doi_norm"] != ""].drop_duplicates(subset=["_doi_norm"], keep="first")
        ref_without_doi = ref_in_sources[ref_in_sources["_doi_norm"] == ""].drop_duplicates(subset=["_title_norm"], keep="first")
        ref_in_sources = pd.concat([ref_with_doi, ref_without_doi], ignore_index=True, sort=False)
    print(f"Reference records present in six-platform sources: {len(ref_in_sources)}")

    desired_min_output = max(len(df_ref) + 20, 120)
    if len(df_final) < desired_min_output:
        existing_titles = set(df_final.get("_title_norm", pd.Series(dtype=str)).dropna().astype(str))
        existing_dois = set(
            df_final.get("_doi_norm", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .loc[lambda s: s != ""]
        )
        rows_to_add = []
        for _, row in df_gate_clean.iterrows():
            title_norm = normalize_text(row.get("_title_norm", "")).strip().lower()
            doi_norm = normalize_text(row.get("_doi_norm", "")).strip().lower()
            if title_norm in existing_titles:
                continue
            if doi_norm and doi_norm in existing_dois:
                continue

            rows_to_add.append(row)
            existing_titles.add(title_norm)
            if doi_norm:
                existing_dois.add(doi_norm)

            if len(df_final) + len(rows_to_add) >= desired_min_output:
                break

        if rows_to_add:
            df_final = pd.concat([df_final, pd.DataFrame(rows_to_add)], ignore_index=True, sort=False)
            print(f"Backfilled from strict gate: +{len(rows_to_add)} rows")

    if "publication_year" in df_final.columns:
        df_final = df_final.sort_values("publication_year", ascending=False)

    final_doi_norm = df_final.get("doi", "").map(normalize_text).str.strip().str.lower()
    final_title_norm = df_final.get("display_name", "").map(normalize_text).str.strip().str.lower()
    in_ref = final_doi_norm.isin(ref_doi_set) | final_title_norm.isin(ref_title_set)
    removed_existing = int(in_ref.sum())
    if removed_existing:
        df_final = df_final[~in_ref].copy()
        print(f"Removed already-existing reference records: {removed_existing}")

    source_norm = df_final.get("_source_folder", "").map(normalize_text).str.strip().str.lower()
    df_final["source_platform"] = source_norm.map(PLATFORM_CANONICAL).fillna("")
    df_final = df_final[df_final["source_platform"].isin(VALID_PLATFORMS)].copy()

    gate_source_norm = df_gate_clean.get("_source_folder", "").map(normalize_text).str.strip().str.lower()
    df_gate_clean["source_platform"] = gate_source_norm.map(PLATFORM_CANONICAL).fillna("")
    df_gate_clean = df_gate_clean[df_gate_clean["source_platform"].isin(VALID_PLATFORMS)].copy()

    gate_in_ref = df_gate_clean["_doi_norm"].isin(ref_doi_set) | df_gate_clean["_title_norm"].isin(ref_title_set)
    df_gate_clean = df_gate_clean[~gate_in_ref].copy()

    desired_min_output = max(len(df_ref) + 100, 180)
    if len(df_final) < desired_min_output:
        existing_titles = set(df_final.get("display_name", pd.Series(dtype=str)).map(normalize_text).str.strip().str.lower())
        existing_dois = set(
            df_final.get("doi", pd.Series(dtype=str))
            .map(normalize_text)
            .str.strip()
            .str.lower()
            .loc[lambda s: s != ""]
        )
        rows_to_add = []
        for _, row in df_gate_clean.sort_values(["_final_score", "_ai_score", "_cite_num"], ascending=[False, False, False]).iterrows():
            title_norm = normalize_text(row.get("_title_norm", "")).strip().lower()
            doi_norm = normalize_text(row.get("_doi_norm", "")).strip().lower()
            if title_norm in existing_titles:
                continue
            if doi_norm and doi_norm in existing_dois:
                continue

            rows_to_add.append(row)
            existing_titles.add(title_norm)
            if doi_norm:
                existing_dois.add(doi_norm)

            if len(df_final) + len(rows_to_add) >= desired_min_output:
                break

        if rows_to_add:
            df_final = pd.concat([df_final, pd.DataFrame(rows_to_add)], ignore_index=True, sort=False)
            print(f"Post-filter source backfill: +{len(rows_to_add)} rows")

    # Add selected reference rows that are found in the six source platforms.
    if not ref_in_sources.empty:
        ref_source_norm = ref_in_sources.get("_source_folder", "").map(normalize_text).str.strip().str.lower()
        ref_in_sources["source_platform"] = ref_source_norm.map(PLATFORM_CANONICAL).fillna("")
        ref_in_sources = ref_in_sources[ref_in_sources["source_platform"].isin(VALID_PLATFORMS)].copy()

        existing_titles = set(df_final.get("display_name", pd.Series(dtype=str)).map(normalize_text).str.strip().str.lower())
        existing_dois = set(
            df_final.get("doi", pd.Series(dtype=str))
            .map(normalize_text)
            .str.strip()
            .str.lower()
            .loc[lambda s: s != ""]
        )

        add_ref_rows = []
        for _, row in ref_in_sources.iterrows():
            title_norm = normalize_text(row.get("_title_norm", "")).strip().lower()
            doi_norm = normalize_text(row.get("_doi_norm", "")).strip().lower()
            if title_norm in existing_titles:
                continue
            if doi_norm and doi_norm in existing_dois:
                continue

            add_ref_rows.append(row)
            existing_titles.add(title_norm)
            if doi_norm:
                existing_dois.add(doi_norm)

        if add_ref_rows:
            df_final = pd.concat([df_final, pd.DataFrame(add_ref_rows)], ignore_index=True, sort=False)
            print(f"Added reference-in-source rows: +{len(add_ref_rows)}")

    core_cols = [
        "id",
        "link",
        "display_name",
        "doi",
        "doi_link",
        "publication_date",
        "publication_year",
        "type",
        "cited_by_count",
        "abstract",
        "keywords",
        "authors",
        "affiliations",
        "countries",
    ]

    drop_output_cols = {
        "source",
        "source_platform",
        "_source_folder",
        "source_file",
        "_ai_score",
        "_ref_sim",
        "_final_score",
    }

    for col in core_cols:
        if col not in df_final.columns:
            df_final[col] = ""

    # Keep model-critical columns first, then include every remaining source column.
    extra_cols = [
        c
        for c in df_final.columns
        if c not in core_cols and c not in drop_output_cols and not c.startswith("_")
    ]
    output_cols = core_cols + extra_cols

    out = df_final[output_cols].copy().reset_index(drop=True)
    out.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved clean output: {OUTPUT_FILE} ({len(out)} rows)")


if __name__ == "__main__":
    main()
