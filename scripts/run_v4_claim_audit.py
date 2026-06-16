from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr2026"
FINAL_DIR = ROOT / "paper" / "final"
FINAL_PDF = FINAL_DIR / "vla-v4.pdf"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
DESKTOP_PDF = DESKTOP / "vla-v4.pdf"
OLD_DESKTOP_PDFS = [DESKTOP / "vla-v3.pdf", DESKTOP / "vla-v2.pdf"]
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SOURCE_ROW = "| `vla-v4.pdf` | `C:\\Users\\wangz\\vla` | `Jason-Wang313/vla` |"
MIN_PAGES = 25


def fail(message: str) -> None:
    raise SystemExit(f"v4 claim audit failed: {message}")


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


def pdf_text(path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", str(path), "-"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.lower()


def normalized(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def contains_marker(text: str, phrase: str) -> bool:
    return phrase in text or normalized(phrase) in normalized(text)


def read_texts() -> str:
    files = [PAPER / "main.tex"]
    files.extend(sorted(PAPER.glob("v3_*.tex")))
    files.extend(sorted(PAPER.glob("v4_*.tex")))
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files)


def main() -> None:
    subprocess.run(["python", "experiments/v4_cached_evidence.py"], cwd=ROOT, check=True)

    for path in [FINAL_PDF, DESKTOP_PDF, SOURCE_MAP]:
        if not path.exists():
            fail(f"missing required artifact {path}")
    for old_pdf in OLD_DESKTOP_PDFS:
        if old_pdf.exists():
            fail(f"old Desktop PDF still exists: {old_pdf}")

    pages = pdf_pages(FINAL_PDF)
    if pages < MIN_PAGES:
        fail(f"final PDF has {pages} pages, expected at least {MIN_PAGES}")
    if pdf_pages(DESKTOP_PDF) != pages:
        fail("Desktop PDF page count differs from repo PDF")

    repo_hash = sha256(FINAL_PDF)
    desktop_hash = sha256(DESKTOP_PDF)
    if repo_hash != desktop_hash:
        fail("repo and Desktop PDF SHA-256 hashes differ")

    manifest_path = FINAL_DIR / "vla_v4_manifest.json"
    if not manifest_path.exists():
        fail("missing v4 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != repo_hash or manifest.get("pages") != pages:
        fail("manifest hash/page data does not match final PDF")
    if manifest.get("source_map_row") != SOURCE_ROW:
        fail("manifest source-map row is wrong")

    source_map = SOURCE_MAP.read_text(encoding="utf-8")
    if SOURCE_ROW not in source_map:
        fail("source map does not contain exact vla-v4 row")
    for old_row in [
        "| `vla-v3.pdf` | `C:\\Users\\wangz\\vla` | `Jason-Wang313/vla` |",
        "| `vla-v2.pdf` | `C:\\Users\\wangz\\vla` | `Jason-Wang313/vla` |",
    ]:
        if old_row in source_map:
            fail(f"source map still contains old row: {old_row}")

    summary_path = ROOT / "results" / "v4_vla_cached_evidence_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("paper_identity") != "VLA semantic affordance over-selection and Certified TailGuard":
        fail("wrong v4 evidence identity")
    if summary.get("version") != "v4" or not summary.get("uses_cached_artifacts_only"):
        fail("v4 summary must be cached v4 evidence")
    if summary.get("target_page_count_minimum") != MIN_PAGES:
        fail("wrong target page count in v4 summary")
    if summary.get("scorecard_rows") != 12:
        fail("v4 scorecard row count is wrong")
    if summary.get("protocol_gates") != 12:
        fail("v4 protocol gate count is wrong")
    if summary.get("rubric_axes") != 7:
        fail("v4 rubric axis count is wrong")
    if summary.get("attack_rounds") != 60:
        fail("v4 attack count is wrong")

    generated = summary.get("generated_artifacts", [])
    for rel in generated:
        if not (ROOT / rel).exists():
            fail(f"generated artifact listed in summary is missing: {rel}")

    with (ROOT / "results" / "v4_reviewer_attack_ledger.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        fail(f"v4 attack ledger has {len(rows)} rows instead of 60")
    statuses = {row["status"] for row in rows}
    if not statuses <= {"pass", "bounded"}:
        fail(f"unexpected v4 attack-ledger statuses: {sorted(statuses)}")
    required_angles = {"protocol", "rubric", "source firewall", "baseline pressure", "stress pressure", "external boundary", "layout", "citations", "claim gate", "remote"}
    angles = {row["reviewer_angle"] for row in rows}
    if not required_angles <= angles:
        fail(f"v4 attack ledger missing reviewer angles: {sorted(required_angles - angles)}")

    text = read_texts().lower()
    pdf_lower = pdf_text(FINAL_PDF)
    duplicate_risk = ["best-of-n", "best of n", "wam", "jepa", "diffusion policy", "world model wrapper"]
    found = [phrase for phrase in duplicate_risk if phrase in text or contains_marker(pdf_lower, phrase)]
    if found:
        fail(f"duplicate-risk phrases found in manuscript/generated tables/PDF: {found}")

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
    found_positive = [phrase for phrase in banned_positive if phrase in text or contains_marker(pdf_lower, phrase)]
    if found_positive:
        fail(f"positive overclaim phrases found: {found_positive}")

    required_boundaries = [
        "no real-robot validation",
        "no physical success",
        "integration status only",
        "action\\_decode\\_supported=false",
        "tailguard\\_method\\_success=false",
        "not a robotics benchmark paper",
        "controlled vla selected-tail audit",
    ]
    missing_boundaries = [phrase for phrase in required_boundaries if phrase not in text]
    if missing_boundaries:
        fail(f"required boundary phrases missing: {missing_boundaries}")

    pdf_required = [
        "v4 vla submission scorecard",
        "v4 protocol-freeze gates",
        "iclr-style rubric map",
        "sixty-round v4 reviewer attack ledger",
        "source firewall",
    ]
    missing_pdf = [phrase for phrase in pdf_required if not contains_marker(pdf_lower, phrase)]
    if missing_pdf:
        fail(f"required PDF markers missing: {missing_pdf}")

    if text.count(r"\citep") < 8:
        fail("citation surface is too sparse")

    term_minima = {
        "vla": 25,
        "semantic": 45,
        "action": 40,
        "physical": 40,
        "certificate": 20,
        "tailguard": 40,
        "calibration": 12,
        "pilot": 12,
        "external": 12,
        "smolvla": 6,
    }
    too_low = {term: text.count(term) for term, minimum in term_minima.items() if text.count(term) < minimum}
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
