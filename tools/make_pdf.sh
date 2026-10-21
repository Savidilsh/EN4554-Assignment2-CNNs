#!/usr/bin/env bash
# Export the executed notebook to HTML, wrap long code/output lines, and print it to PDF with headless Edge.
set -e
cd "$(dirname "$0")/.."
NAME=${1:-220701X_a02}
.venv/Scripts/python.exe -m jupyter nbconvert --to html a02_cnn.ipynb --output "$NAME"
.venv/Scripts/python.exe - "$NAME.html" <<'PY'
import sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
css = ("<style>@media print { .jp-Cell { break-inside: auto; } } "
       "pre, .jp-OutputArea-output pre, .highlight pre { white-space: pre-wrap !important; word-break: break-word; } "
       ".jp-RenderedHTMLCommon table { font-size: 11px; } img { max-width: 100% !important; }</style>")
open(p, "w", encoding="utf-8").write(s.replace("</head>", css + "</head>", 1))
PY
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
rm -f "$NAME.pdf"  # Edge does not overwrite an existing file
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=30000 --user-data-dir="$(cygpath -w "$TEMP")\\edgepdf" \
  --print-to-pdf="$(cygpath -w "$PWD/$NAME.pdf")" "file:///$(cygpath -m "$PWD")/$NAME.html" 2>&1 | grep -v "sync\|GetUpdates\|task_provider" || true
ls -la "$NAME.pdf"
