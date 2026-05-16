#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

out="${1:-closed_loop_rnn_audit_anonymous_supplement.tar.gz}"
prefix="closed_loop_rnn_audit_anonymous/"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to build from a dirty worktree. Commit or stash changes first." >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

git archive --format=tar --prefix="$prefix" HEAD | tar -xf - -C "$tmpdir"

if find "$tmpdir/$prefix" -name .git -o -name .venv -o -name __pycache__ | grep -q .; then
  echo "Anonymous archive staging area contains excluded local state." >&2
  exit 1
fi

leak_patterns=(
  "a""arav"
  "a""aravsinha"
  "a""aravsinhaofficial"
  "a""aravofficial009"
  "github.com/a""aravsinha"
  "software""heritage.org/browse/origin"
  "origin""_url="
  "/Users/a""arav"
  "/home/""ubuntu"
  "ubuntu""@"
  "ip""-172-"
  "reproduce""_tmlr"
)

for pattern in "${leak_patterns[@]}"; do
  if rg -I -n -i --glob '!external/original_artifact/**' "$pattern" "$tmpdir/$prefix" >/tmp/anonymous_supplement_leaks.txt; then
    echo "Potential anonymity leak for pattern: $pattern" >&2
    cat /tmp/anonymous_supplement_leaks.txt >&2
    exit 1
  fi
done

if command -v python >/dev/null 2>&1; then
  python - "$tmpdir/$prefix/paper/closed_loop_rnn_audit_tmlr.pdf" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)

try:
    from PyPDF2 import PdfReader
except Exception:
    raise SystemExit(0)

reader = PdfReader(str(path))
metadata = " ".join(str(v) for v in (reader.metadata or {}).values()).lower()
text = "\n".join((page.extract_text() or "") for page in reader.pages).lower()
patterns = [
    "a" "arav",
    "a" "aravsinha",
    "a" "aravsinhaofficial",
    "a" "aravofficial009",
    "github.com/a" "aravsinha",
    "software" "heritage.org/browse/origin",
    "origin" "_url=",
    "/users/a" "arav",
    "/home/" "ubuntu",
    "ubuntu" "@",
    "ip" "-172-",
    "reproduce" "_tmlr",
]
for pattern in patterns:
    if pattern in metadata or pattern in text:
        raise SystemExit(f"Potential PDF anonymity leak for pattern: {pattern}")
PY
fi

rm -f "$out"
tar -czf "$out" -C "$tmpdir" "$prefix"

echo "wrote $out"
echo "contents are generated from tracked files only; .git, remotes, commit history, local logs, and untracked result bundles are excluded"
