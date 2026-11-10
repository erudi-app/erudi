#!/usr/bin/env python3
"""Fail when a frozen backend cannot resolve python-docx's default parts.

Usage: python scripts/ci/check_docx_default_parts.py <dir>
       python scripts/ci/check_docx_default_parts.py backend/dist

python-docx builds the path of each default part (styles, settings, header,
footer, comments) as ``<docx/parts>/../templates/<name>``. A ``.docx`` saved
without a styles part (TextEdit does) reaches it. In a PyInstaller bundle the
modules of ``docx.parts`` live in the PYZ archive, so unless the spec collects
``docx`` as sources too, ``docx/parts/`` is not a directory and the OS cannot
resolve the ``..``: macOS answers ENOENT and ingestion of the file fails (#705).

The check walks <dir>, finds every ``docx/templates`` directory and requires
that the five default parts are there and that the sibling ``docx/parts`` is a
real directory, then resolves each part through the very path python-docx
builds. It does not start the frozen app: it checks the one thing the bundle
decides, and costs nothing on any OS.

Exit codes: 0 every docx tree resolves; 1 one does not, or none was found (an
empty walk proves nothing); 2 usage error or <dir> is not a directory.
"""

import os
import sys
from pathlib import Path

# The files python-docx ships in docx/templates and reads by this path.
DEFAULT_PARTS = (
    "default-comments.xml",
    "default-footer.xml",
    "default-header.xml",
    "default-settings.xml",
    "default-styles.xml",
)


def _problems(docx_dir: Path) -> list[str]:
    parts = docx_dir / "parts"
    if not parts.is_dir():
        return [f"{parts} is not a directory, so '..' cannot resolve through it"]
    # The exact construction of docx/parts/styles.py and its siblings.
    return [
        f"{os.path.join(parts, '..', 'templates', name)} does not resolve"
        for name in DEFAULT_PARTS
        if not os.path.isfile(os.path.join(parts, "..", "templates", name))
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_docx_default_parts.py <dir>", flush=True)
        return 2
    root = Path(argv[0])
    if not root.is_dir():
        print(f"CHECK FAIL: {root} is not a directory", flush=True)
        return 2

    trees = sorted(p.parent for p in root.rglob("templates") if p.parent.name == "docx")
    if not trees:
        print(f"CHECK FAIL: no docx package found under {root}", flush=True)
        return 1

    failed = False
    for docx_dir in trees:
        problems = _problems(docx_dir)
        for problem in problems:
            print(f"CHECK FAIL: {problem}", flush=True)
        failed = failed or bool(problems)
        if not problems:
            print(f"CHECK OK: {docx_dir} resolves its default parts", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
