"""Build the static site: source/template.html + data/cases.json -> site/index.html

Standard-library Python only. Run from anywhere:  python website/source/build_site.py
Images are not touched: they already live in site/img/ and cases.json points at them.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # website/
TEMPLATE = ROOT / "source" / "template.html"
DATA = ROOT / "data" / "cases.json"
SITE = ROOT / "site"


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    missing = [p for c in data["cases"] for p in [*c["views"]["axial"], c["views"]["coronal"], c["views"]["sagittal"]]
               if not (SITE / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images referenced in cases.json are missing from site/, e.g. {missing[:3]}")

    tpl = TEMPLATE.read_text(encoding="utf-8")
    head, body = tpl.split("</style>", 1)
    head = re.sub(r'<meta name="claude:viewport-fit-sides"[^>]*>\n?', "", head)
    body = body.replace("__DEMO_DATA__", json.dumps(data).replace("</", "<\\/"))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Real MRI cases from the iAudit project with every brain-age model's prediction shown against the true age.">
<style>
:root {{ padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px) }}
body {{ margin: 0 }} img {{ max-width: 100% }} [hidden] {{ display: none !important }}
</style>
{head}</style>
</head>
<body>
{body}
</body>
</html>
"""
    assert "__DEMO_DATA__" not in page
    (SITE / "index.html").write_text(page, encoding="utf-8")
    print(f"built {SITE / 'index.html'} ({len(data['cases'])} cases)")


if __name__ == "__main__":
    main()
