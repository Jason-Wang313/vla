from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr2026"
MAIN_PDF = PAPER / "main.pdf"
FINAL_DIR = ROOT / "paper" / "final"
FINAL_PDF = FINAL_DIR / "vla-v4.pdf"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
DESKTOP_PDF = DESKTOP / "vla-v4.pdf"
OLD_DESKTOP_PDFS = [DESKTOP / "vla-v3.pdf", DESKTOP / "vla-v2.pdf"]
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SOURCE_ROW = "| `vla-v4.pdf` | `C:\\Users\\wangz\\vla` | `Jason-Wang313/vla` |"
MIN_PAGES = 25


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


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
    raise RuntimeError(f"could not read page count from {path}")


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


def update_source_map() -> None:
    text = SOURCE_MAP.read_text(encoding="utf-8")
    if "## V4 Verification Ledger" in text:
        lookup_text, ledger_text = text.split("## V4 Verification Ledger", 1)
        ledger_text = "## V4 Verification Ledger" + ledger_text
    else:
        lookup_text, ledger_text = text, ""
    lines = lookup_text.splitlines()
    replaced = False
    out: list[str] = []
    for line in lines:
        if "| `vla-" in line and "`C:\\Users\\wangz\\vla`" in line and "`Jason-Wang313/vla`" in line:
            out.append(SOURCE_ROW)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise RuntimeError("could not find existing vla source-map row")
    updated = "\n".join(out).rstrip() + "\n"
    if ledger_text:
        updated += "\n" + ledger_text.rstrip() + "\n"
    SOURCE_MAP.write_text(updated, encoding="utf-8")


def validate_pdf_text() -> None:
    text = pdf_text(MAIN_PDF)
    required = [
        "semantic affordance over-selection",
        "certified tailguard",
        "v4 vla submission scorecard",
        "v4 protocol-freeze gates",
        "iclr-style rubric map",
        "sixty-round v4 reviewer attack ledger",
        "source firewall",
        "integration status only",
        "no real-robot validation",
        "no physical success",
        "action_decode_supported=false",
        "tailguard_method_success=false",
    ]
    missing = [phrase for phrase in required if not contains_marker(text, phrase)]
    if missing:
        raise RuntimeError(f"final PDF missing required v4 markers: {missing}")

    forbidden = [
        "best-of-n",
        "best of n",
        "world model wrapper",
        "we validate on real robots",
        "validated on real robots",
        "real-robot validation is established",
        "external benchmark success is established",
        "tailguard succeeds on robocasa",
        "physical success on robocasa",
        "solves robot planning",
        "proves vlas work",
    ]
    found = [phrase for phrase in forbidden if contains_marker(text, phrase)]
    if found:
        raise RuntimeError(f"final PDF contains forbidden overclaim or duplicate-risk text: {found}")


def main() -> None:
    run(["python", "experiments/v4_cached_evidence.py"])
    run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts/build_iclr_submission.ps1"])

    pages = pdf_pages(MAIN_PDF)
    if pages < MIN_PAGES:
        raise RuntimeError(f"page count {pages} is below required minimum {MIN_PAGES}")
    validate_pdf_text()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MAIN_PDF, FINAL_PDF)
    shutil.copy2(MAIN_PDF, DESKTOP_PDF)
    for old_pdf in OLD_DESKTOP_PDFS:
        if old_pdf.exists():
            old_pdf.unlink()
    update_source_map()

    repo_hash = sha256(FINAL_PDF)
    desktop_hash = sha256(DESKTOP_PDF)
    if repo_hash != desktop_hash:
        raise RuntimeError("repo and Desktop final PDFs differ")

    manifest = {
        "paper": "vla",
        "version": "v4",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "min_pages": MIN_PAGES,
        "pages": pages,
        "repo_pdf": str(FINAL_PDF),
        "desktop_pdf": str(DESKTOP_PDF),
        "sha256": repo_hash,
        "source_map": str(SOURCE_MAP),
        "source_map_row": SOURCE_ROW,
    }
    (FINAL_DIR / "vla_v4_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
