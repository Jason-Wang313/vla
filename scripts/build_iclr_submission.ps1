$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$paperDir = Join-Path $repoRoot "paper\iclr2026"
$source = Join-Path $paperDir "main.tex"
$builtPdf = Join-Path $paperDir "main.pdf"
$downloads = Join-Path $env:USERPROFILE "Downloads"
$destination = Join-Path $downloads "certified-tailguard-vla-iclr-submission.pdf"

if (-not (Test-Path -LiteralPath $source)) {
    throw "Cannot find LaTeX source: $source"
}

New-Item -ItemType Directory -Force -Path $downloads | Out-Null

Push-Location $paperDir
try {
    function Invoke-PdfLatexFallback {
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed on first pass" }
        & bibtex main
        if ($LASTEXITCODE -ne 0) { throw "bibtex failed" }
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed on second pass" }
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed on final pass" }
    }

    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
    if ($latexmk) {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $builtPdf)) {
            Write-Warning "latexmk failed or produced no PDF; falling back to pdflatex/bibtex."
            Invoke-PdfLatexFallback
        }
    } else {
        Invoke-PdfLatexFallback
    }
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $builtPdf)) {
    throw "LaTeX did not produce $builtPdf"
}

Copy-Item -LiteralPath $builtPdf -Destination $destination -Force
$size = (Get-Item -LiteralPath $destination).Length
Write-Host "Built submission PDF: $destination ($size bytes)"
