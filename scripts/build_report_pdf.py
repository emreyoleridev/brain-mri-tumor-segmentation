"""Compile report/report.typ to REPORT.pdf with the Typst compiler (pip install typst)."""
from pathlib import Path

import typst

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    out = ROOT / "REPORT.pdf"
    typst.compile(str(ROOT / "report" / "report.typ"), output=str(out), root=str(ROOT))
    print("Wrote", out)
