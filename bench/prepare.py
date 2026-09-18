"""Download pinned public benchmark sources and build the common parquet layout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import struct
import sys
import zipfile
import zlib

import httpx
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import DATA, NAMES, write_dataset

LEIPZIG_PAGE = "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution"
LEIPZIG = {
    "abt-buy": ("Abt-Buy.zip", "b9ff2937d97371b7a00b0a97033213bfbdee24e8f1a0718a41ea8364472a8ff8",
                "Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv"),
    "dblp-acm": ("DBLP-ACM.zip", "e37e7ed8e06722499e6d9e94583fdec279207b13ce6270a37595b6a5d594bc40",
                 "DBLP2.csv", "ACM.csv", "DBLP-ACM_perfectMapping.csv"),
    "amazon-google": ("Amazon-GoogleProducts.zip",
                      "fcd72e223d51ce5b0d6b1d68034a438ba95cf2940431e209a57910b067cc4811",
                      "Amazon.csv", "GoogleProducts.csv", "Amzon_GoogleProducts_perfectMapping.csv"),
}
NBER_URL = "https://data.nber.org/patents/amatch.zip"
NBER_SHA256 = "98d89982c9d09c3b07cb5f92b98e669d4afc22b9b1d71d8d57e1c7acaa017cfe"
FEBRL_HASHES = {
    "dataset4a.csv": "07c7cb3f0a8d88180e80317f2a60499dee4e8324a44c38059f4e7fed0a8b4488",
    "dataset4b.csv": "2eed76c99fa2237be3ec013a123427926d4158abcb3a8f65874d6c7f1358cf2c",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def download(url: str, checksum: str, directory: Path) -> tuple[Path, dict]:
    """Cache a verified source with its actual download timestamp; reject upstream drift."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / url.rsplit("/", 1)[-1]
    receipt = path.with_suffix(path.suffix + ".json")
    if path.exists() and receipt.exists():
        info = json.loads(receipt.read_text())
        if _sha256(path.read_bytes()) != checksum:
            raise ValueError(f"cached source {path} failed SHA-256 verification; remove it and prepare again")
        if info.get("url") != url or info.get("sha256") != checksum:
            raise ValueError(f"download receipt {receipt} does not describe the expected source")
        return path, info
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    content = response.content
    if _sha256(content) != checksum:
        raise ValueError(f"source {url} changed (SHA-256 differs); "
                         "inspect the new release before updating the pin")
    info = {"url": url, "resolved_url": str(response.url), "sha256": checksum,
            "downloaded_at": _now(), "bytes": len(content)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    receipt.write_text(json.dumps(info, indent=2) + "\n")
    return path, info


def _csv(content: bytes) -> pd.DataFrame:
    # Leipzig's older product files contain Windows-1252 characters; never replace undecodable bytes.
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1252")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _leipzig(name: str, downloads: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    archive, checksum, left_file, right_file, truth_file = LEIPZIG[name]
    url = f"https://dbs.uni-leipzig.de/files/datasets/{archive}"
    path, source = download(url, checksum, downloads)
    with zipfile.ZipFile(path) as z:
        left, right, truth = [_csv(z.read(member)) for member in (left_file, right_file, truth_file)]
    truth.columns = ["left_id", "right_id"]
    if name == "dblp-acm":
        entity, on = "publication", ["title", "authors", "year"]
        definition = ("Two bibliographic records match when they describe the same published paper. "
                      "A different paper by the same authors or with a similar title is not a match.")
        how = "one-to-one"
    else:
        entity = "product"
        on = ["name", "description", "price"] if name == "abt-buy" else ["name", "manufacturer", "price"]
        left = left.rename(columns={"title": "name"})
        definition = ("Two listings match when they offer the same product model. The same product line "
                      "in a different size, color, capacity, software version or license bundle "
                      "is not a match.")
        how = "many-to-many"
    meta = {"entity": entity, "definition": definition, "on": on, "how": how,
            "source_url": url, "source_page": LEIPZIG_PAGE,
            "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Kopcke, Thor and Rahm (2010), Evaluation of entity resolution approaches, VLDB.",
            "sources": [source], "download_date": source["downloaded_at"][:10],
            "source_row_counts": {"left": len(left), "right": len(right), "truth": len(truth)},
            "notes": ["Use the complete original Leipzig mapping, not a sampled train/test candidate set.",
                      "Unlisted pairs are treated as nonmatches for benchmark scoring."]}
    return left, right, truth, meta


def _febrl() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    import recordlinkage
    from recordlinkage.datasets import load_febrl4

    version = importlib.metadata.version("recordlinkage")
    root = Path(recordlinkage.__file__).parent / "datasets" / "febrl"
    sources = []
    for filename, expected in FEBRL_HASHES.items():
        checksum = _sha256((root / filename).read_bytes())
        if checksum != expected:
            raise ValueError(f"recordlinkage's {filename} changed; "
                             "use the version in uv.lock or inspect the release")
        sources.append({"url": f"https://github.com/J535D165/recordlinkage/blob/v{version}/"
                               f"recordlinkage/datasets/febrl/{filename}", "sha256": checksum,
                        "acquired_at": _now(), "package_version": version})
    left, right, links = load_febrl4(return_links=True)
    left, right = [frame.rename_axis("id").reset_index() for frame in (left, right)]
    truth = links.to_frame(index=False, name=["left_id", "right_id"])
    meta = {
        "entity": "person", "definition": "Two records match when they were generated from the same "
        "underlying person in FEBRL, despite errors or missing identifying information. "
        "Sharing a surname or an address alone does not establish a match.",
        "on": ["given_name", "surname", "date_of_birth", "address_1", "suburb", "postcode"],
        "how": "one-to-one", "source_url": "https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html",
        "license": "recordlinkage distribution: BSD-3-Clause; generated FEBRL example data bundled therein",
        "license_url": "https://github.com/J535D165/recordlinkage/blob/v0.16/LICENSE",
        "sources": sources, "download_date": sources[0]["acquired_at"][:10],
        "source_row_counts": {"left": len(left), "right": len(right), "truth": len(truth)},
        "notes": ["Synthetic people, not observations of real individuals.",
                  "Downloaded with recordlinkage by uv sync --group bench. The date records acquisition "
                  "from the installed package, not its earlier installation or upstream publication date."]}
    return left, right, truth, meta


def _nber_csv(path: Path) -> bytes:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.read("match.csv")
    except zipfile.BadZipFile as exc:
        if "Corrupt extra field" not in str(exc):
            raise
        # The pinned 2002 ZIP has malformed extra metadata, but a valid single-member local header.
        # Read that member using the documented ZIP layout, then verify length and CRC independently.
        data = path.read_bytes()
        if _sha256(data) != NBER_SHA256:
            raise ValueError("legacy ZIP fallback is only supported for the pinned NBER archive") from exc
        signature, version, flags, method, time, date, crc, size, length, name_size, extra_size = \
            struct.unpack_from("<4s5H3I2H", data)
        if signature != b"PK\x03\x04" or flags != 0 or method != 8 or data[30:30 + name_size] != b"match.csv":
            raise ValueError("NBER ZIP has an unexpected local member header") from exc
        start = 30 + name_size + extra_size
        content = zlib.decompress(data[start:start + size], wbits=-15)
        if len(content) != length or zlib.crc32(content) != crc:
            raise ValueError("NBER ZIP member failed size or CRC verification") from exc
        return content


def build_nber(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Keep labeled historical associations; collapse name variants without exposing IDs as features."""
    from jlink.fields import check_columns

    check_columns(raw, ["assignee", "assname", "cusip", "cname"], "NBER source")
    complete = raw[["assignee", "assname", "cusip", "cname"]].notna().all(axis=1)
    complete &= raw[["assignee", "assname", "cusip", "cname"]].ne("").all(axis=1)
    matched = raw.loc[complete].copy()
    if matched.empty:
        raise ValueError("the NBER source has no complete labeled assignee, name, CUSIP "
                         "and company-name rows")

    def names(id_col: str, name_col: str) -> pd.DataFrame:
        # Most frequent source spelling, then lexical tie-breaking; aliases remain available for inspection.
        counts = matched.groupby([id_col, name_col]).size().rename("count").reset_index()
        canonical = counts.sort_values([id_col, "count", name_col], ascending=[True, False, True])
        canonical = canonical.drop_duplicates(id_col).rename(columns={id_col: "id", name_col: "name"})
        aliases = matched.groupby(id_col)[name_col].agg(lambda values: " | ".join(sorted(set(values))))
        canonical["source_names"] = canonical.id.map(aliases)
        return canonical[["id", "name", "source_names"]].reset_index(drop=True)

    left, right = names("assignee", "assname"), names("cusip", "cname")
    truth = matched[["assignee", "cusip"]].rename(columns={"assignee": "left_id", "cusip": "right_id"})
    diagnostics = {
        "source_rows": len(raw), "source_duplicate_rows": int(raw.duplicated().sum()),
        "source_repeated_assignee_rows": int(raw.assignee.duplicated().sum()),
        "excluded_incomplete_rows": int((~complete).sum()),
        "complete_rows_before_pair_deduplication": len(matched),
        "assignees_with_multiple_names": int(matched.groupby("assignee").assname.nunique().gt(1).sum()),
        "cusips_with_multiple_names": int(matched.groupby("cusip").cname.nunique().gt(1).sum()),
    }
    return left, right, truth, diagnostics


def _nber(downloads: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    path, source = download(NBER_URL, NBER_SHA256, downloads)
    left, right, truth, diagnostics = build_nber(_csv(_nber_csv(path)))
    meta = {
        "entity": "firm", "definition": "A patent assignee matches a Compustat company when the historical "
        "NBER assignee-to-Compustat crosswalk assigns it to that company, "
        "including subsidiary-to-parent links. "
        "This is the crosswalk's historical corporate association, not necessarily the same legal entity "
        "or its present owner.",
        "on": ["name"], "how": "many-to-many", "source_url": NBER_URL,
        "source_page": "https://www.nber.org/research/data/us-patents-1975-1999",
        "documentation_url": "https://data.nber.org/patents/match.txt",
        "license": "Publicly downloadable NBER research data; no explicit redistribution license found "
                   "on the data page or match.txt. Underlying sources include USPTO and Compustat.",
        "citation": "Hall, Jaffe and Trajtenberg (2001), The NBER Patent Citation Data File, NBER WP 8498.",
        "sources": [source], "download_date": source["downloaded_at"][:10],
        "source_row_counts": {"crosswalk": diagnostics["source_rows"]}, "construction": diagnostics,
        "notes": ["Original 1975-1999 NBER release; this is not the later dynamic PDP match.",
                  "Keep only rows with both identifiers and both names. Missing labels are not negatives.",
                  "One record per assignee and per CUSIP; choose the most frequent name, then lexical order.",
                  "Only name is a matching field: identifiers, parent names "
                  "and ownership labels are withheld.",
                  "The search universe contains labeled firms only; "
                  "there are no known unmatched left records.",
                  "Unlisted pairs are assumed negative, although historical labels may be incomplete.",
                  "CUSIPs and parent ownership can change over time; no date-specific ownership is inferred.",
                  "The pinned legacy ZIP is decoded with the standard library and checked by SHA-256, "
                  "uncompressed length and CRC because zipfile rejects its malformed extra metadata."]}
    return left, right, truth, meta


def prepare(name: str, data_dir: Path = DATA) -> dict:
    """Prepare one named dataset from public sources, with no generated replacement data."""
    if name not in NAMES:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(NAMES)}")
    downloads = Path(data_dir) / "_downloads"
    if name == "febrl4":
        left, right, truth, meta = _febrl()
    elif name == "nber-firms":
        left, right, truth, meta = _nber(downloads)
    else:
        left, right, truth, meta = _leipzig(name, downloads)
    meta.update(name=name, prepared_at=_now(), schema_version=1)
    return write_dataset(Path(data_dir) / name, left, right, truth, meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="NAME", help=f"default: all ({', '.join(NAMES)})")
    args = parser.parse_args(argv)
    failures = []
    for name in args.names or NAMES:
        try:
            meta = prepare(name)
            print(f"{name}: {json.dumps(meta['row_counts'])}; validation={json.dumps(meta['validation'])}")
        except (ValueError, OSError, httpx.HTTPError, zipfile.BadZipFile, zlib.error) as exc:
            print(f"prepare: {name}: {exc}", file=sys.stderr)
            failures.append(name)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
