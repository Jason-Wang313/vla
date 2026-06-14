from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr2026"
FINAL_DIR = ROOT / "paper" / "final"
FINAL_PDF = FINAL_DIR / "vla-v3.pdf"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
DESKTOP_PDF = DESKTOP / "vla-v3.pdf"
OLD_DESKTOP_PDF = DESKTOP / "vla-v2.pdf"
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SOURCE_ROW = "| `vla-v3.pdf` | `C:\\Users\\wangz\\vla` | `Jason-Wang313/vla` |"
MIN_PAGES = 25


def fail(message: str) -> None:
    raise SystemExit(f"v3 claim audit failed: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    proc = subprocess.run(
        ["pdfinfo", str(path)],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    fail(f"could not read page count from {path}")


def read_texts() -> str:
    files = [PAPER / "main.tex"]
    files.extend(sorted(PAPER.glob("v3_*.tex")))
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files)


def main() -> None:
    for path in [FINAL_PDF, DESKTOP_PDF, SOURCE_MAP]:
        if not path.exists():
            fail(f"missing required artifact {path}")
    if OLD_DESKTOP_PDF.exists():
        fail(f"old Desktop PDF still exists: {OLD_DESKTOP_PDF}")

    pages = pdf_pages(FINAL_PDF)
    if pages < MIN_PAGES:
        fail(f"final PDF has {pages} pages, expected at least {MIN_PAGES}")
    if pdf_pages(DESKTOP_PDF) != pages:
        fail("Desktop PDF page count differs from repo PDF")

    repo_hash = sha256(FINAL_PDF)
    desktop_hash = sha256(DESKTOP_PDF)
    if repo_hash != desktop_hash:
        fail("repo and Desktop PDF SHA-256 hashes differ")

    manifest_path = FINAL_DIR / "vla_v3_manifest.json"
    if not manifest_path.exists():
        fail("missing v3 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != repo_hash or manifest.get("pages") != pages:
        fail("manifest hash/page data does not match final PDF")
    if manifest.get("source_map_row") != SOURCE_ROW:
        fail("manifest source-map row is wrong")

    source_map = SOURCE_MAP.read_text(encoding="utf-8")
    if SOURCE_ROW not in source_map:
        fail("source map does not contain exact vla-v3 row")
    if "| `vla-v2.pdf` | `C:\\Users\\wangz\\vla` | `Jason-Wang313/vla` |" in source_map:
        fail("source map still contains vla-v2 row")

    summary_path = ROOT / "results" / "v3_vla_evidence_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("paper_identity") != "VLA semantic affordance over-selection and Certified TailGuard":
        fail("wrong v3 evidence identity")
    if not summary.get("uses_cached_artifacts_only"):
        fail("v3 summary must use cached full artifacts only")
    if summary.get("target_page_count_minimum") != MIN_PAGES:
        fail("wrong target page count in v3 summary")

    ledger_path = ROOT / "results" / "v3_vla_attack_ledger.csv"
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        fail(f"attack ledger has {len(rows)} rows instead of 50")
    statuses = {row["status"] for row in rows}
    if not statuses <= {"pass", "bounded"}:
        fail(f"unexpected attack-ledger statuses: {sorted(statuses)}")
    required_angles = {"novelty", "scope", "assumption", "robustness", "ablation", "baseline", "external", "workflow"}
    angles = {row["reviewer_angle"] for row in rows}
    if not required_angles <= angles:
        fail(f"attack ledger missing reviewer angles: {sorted(required_angles - angles)}")

    generated = summary.get("generated_artifacts", [])
    for rel in generated:
        if not (ROOT / rel).exists():
            fail(f"generated artifact listed in summary is missing: {rel}")

    text = read_texts()
    lower = text.lower()
    duplicate_risk = ["best-of-n", "best of n", "wam", "jepa", "diffusion policy", "world model wrapper"]
    found = [phrase for phrase in duplicate_risk if phrase in lower]
    if found:
        fail(f"duplicate-risk phrases found in manuscript/generated tables: {found}")

    banned_positive = [
        "we validate on real robots",
        "validated on real robots",
        "real-robot validation is established",
        "external benchmark success is established",
        "tailguard succeeds on robocasa",
        "physical success on robocasa",
        "solves robot planning",
        "proves vlas work",
    ]
    found_positive = [phrase for phrase in banned_positive if phrase in lower]
    if found_positive:
        fail(f"positive overclaim phrases found: {found_positive}")

    required_boundaries = [
        "no real-robot validation",
        "no physical success",
        "integration status only",
        "action\\_decode\\_supported=false",
        "tailguard\\_method\\_success=false",
        "not a robotics benchmark paper",
    ]
    missing_boundaries = [phrase for phrase in required_boundaries if phrase not in lower]
    if missing_boundaries:
        fail(f"required boundary phrases missing: {missing_boundaries}")

    term_minima = {
        "vla": 25,
        "semantic": 40,
        "action": 35,
        "physical": 35,
        "certificate": 15,
        "tailguard": 35,
        "calibration": 10,
        "pilot": 10,
        "external": 10,
        "smolvla": 5,
    }
    too_low = {term: lower.count(term) for term, minimum in term_minima.items() if lower.count(term) < minimum}
    if too_low:
        fail(f"paper-specific term counts too low: {too_low}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "pages": pages,
                "sha256": repo_hash,
                "attack_rounds": len(rows),
                "source_map_row": SOURCE_ROW,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
