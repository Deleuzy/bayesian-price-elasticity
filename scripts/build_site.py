"""Build the static results site that cd.yml publishes to GitHub Pages.

The page is the generated README rendered to HTML, with the figures and the raw
result files next to it, so every number on the page links back to a file in results/.

    uv run --with markdown python scripts/build_site.py --out site
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CSS = """
body { max-width: 900px; margin: 2rem auto; padding: 0 16px; font: 16px/1.6 system-ui, sans-serif;
       color: #1f2328; background: #fff; }
img { max-width: 100%; }
table { border-collapse: collapse; display: block; overflow-x: auto; font-size: 14px; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; }
code, pre { background: #f6f8fa; border-radius: 4px; }
pre { padding: 12px; overflow-x: auto; }
footer { margin-top: 3rem; font-size: 13px; color: #59636e; }
@media (prefers-color-scheme: dark) {
  body { color: #e6edf3; background: #0d1117; }
  th, td { border-color: #3d444d; }
  code, pre { background: #161b22; }
  footer { color: #9198a1; }
}
"""


def commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for d in ("figures", "results"):
        shutil.copytree(ROOT / d, out / d)
    body = markdown.markdown((ROOT / "README.md").read_text(), extensions=["tables", "fenced_code"])
    files = "".join(
        f'<li><a href="results/{p.name}">{p.name}</a></li>'
        for p in sorted((ROOT / "results").iterdir())
        if p.is_file()
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Price elasticity results</title><style>{CSS}</style></head>
<body>{body}
<h2>Result files</h2><ul>{files}</ul>
<footer>Built from commit {commit()} on {stamp} by the cd workflow, after lint, tests and the
README-matches-results check passed.</footer>
</body></html>
"""
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")
    print(f"site written to {out}/index.html")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    build(ap.parse_args(argv).out)


if __name__ == "__main__":
    main()
