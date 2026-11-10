"""scripts/ci/check_docx_default_parts.py: the frozen-bundle check for #705.

python-docx builds the path of its default parts as
``<docx/parts>/../templates/<name>``. The check walks a PyInstaller output tree
and fails unless, next to every ``docx/templates`` that holds the five default
parts, ``docx/parts`` is a real directory (so the OS can resolve the ``..``).
"""

import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_docx_default_parts", _ROOT / "scripts" / "ci" / "check_docx_default_parts.py"
)
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)

pytestmark = pytest.mark.unit


def _bundle(root: pathlib.Path, *, parts: bool, templates=None) -> pathlib.Path:
    docx = root / "backend" / "_internal" / "docx"
    (docx / "templates").mkdir(parents=True)
    for name in templates if templates is not None else check.DEFAULT_PARTS:
        (docx / "templates" / name).write_text("<x/>")
    if parts:
        (docx / "parts").mkdir()
        (docx / "parts" / "styles.py").write_text("")
    return root


def test_passes_when_parts_directory_and_templates_exist(tmp_path):
    assert check.main([str(_bundle(tmp_path, parts=True))]) == 0


def test_fails_when_the_parts_directory_is_missing(tmp_path, capsys):
    # The v1.1.3 bundle: templates on disk, modules only in the PYZ.
    assert check.main([str(_bundle(tmp_path, parts=False))]) == 1
    assert "parts" in capsys.readouterr().out


def test_fails_when_a_default_part_is_missing(tmp_path):
    names = tuple(n for n in check.DEFAULT_PARTS if n != "default-styles.xml")
    assert check.main([str(_bundle(tmp_path, parts=True, templates=names))]) == 1


def test_fails_when_no_docx_package_is_found(tmp_path):
    # An empty walk proves nothing.
    assert check.main([str(tmp_path)]) == 1


def test_fails_on_a_missing_directory(tmp_path):
    assert check.main([str(tmp_path / "absent")]) == 2


def test_the_default_parts_are_the_ones_python_docx_ships():
    import docx

    shipped = {p.name for p in (pathlib.Path(docx.__file__).parent / "templates").glob("*.xml")}
    assert set(check.DEFAULT_PARTS) == shipped
