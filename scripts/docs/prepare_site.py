"""Prepare the built documentation for www.erudi.app, and the redirects of its former host.

The documentation is served at https://www.erudi.app/open-source/desktop/docs by the
website's CDN, whose edge function serves `/x` from `/x/index.html` and redirects
`/x/` to `/x` (301): pages live at URLs without a trailing slash. The theme writes
every link relative to a page URL WITH a trailing slash (`../../assets/...`,
`../llms/`), which a page served at `/docs/guides/hardware` would resolve one level
too high. `rewrite` turns every relative URL into a root path and every page URL
into its clean form, in the HTML, the search index and the sitemap; `check` proves
the result is consistent; `redirects` writes the pages left on the former host
(GitHub Pages, which cannot answer a 301).

Usage (from the repository root, after `zensical build`):
    python scripts/docs/prepare_site.py redirects site pages-redirects \
        --site-url https://www.erudi.app/open-source/desktop/docs --old-base-path /erudi
    python scripts/docs/prepare_site.py rewrite site \
        --site-url https://www.erudi.app/open-source/desktop/docs
    python scripts/docs/prepare_site.py check site \
        --site-url https://www.erudi.app/open-source/desktop/docs

Stdlib only: it runs before and without the documentation tooling.
"""

import argparse
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

# Attributes holding a URL in the theme's output.
URL_ATTRIBUTE = re.compile(r'(\s(?:href|src)=")([^"]*)(")')
# Scope of the theme's local storage keys, resolved against `location`.
SCOPE_SCRIPT = re.compile(r'(new URL\(")([^"]*)(",location\))')
CONFIG_SCRIPT = re.compile(
    r'(<script id="__config" type="application/json">)(.*?)(</script>)', re.S
)
SITEMAP_LOC = re.compile(r"(<loc>)([^<]*)(</loc>)")
# The repository widget asks the GitHub API for stars and forks from the reader's
# browser. Without this attribute the theme leaves it a plain link: the docs make
# no third-party request (the website's privacy policy lists every one).
SOURCE_FACTS = ' data-md-component="source"'
# Third-party hosts and former addresses a page must never reference.
FORBIDDEN = (
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "erudi-app.github.io",
    SOURCE_FACTS.strip(),
)
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def split_site_url(site_url: str) -> tuple[str, str]:
    """("https://host", "/base/path") from the documentation's URL."""
    parts = urlsplit(site_url)
    return f"{parts.scheme}://{parts.netloc}", parts.path.rstrip("/")


def clean_path(path: str, base_path: str) -> str:
    """Clean URL of a page of the documentation; any other path is returned as is."""
    if path != base_path and not path.startswith(base_path + "/"):
        return path
    last = path.rsplit("/", 1)[-1]
    if last == "index.html":
        path = path[: -len(last)]
    elif last:
        return path  # a file (asset, search index) or an already clean page
    return path.rstrip("/") or "/"


def _join(path: str, query: str, fragment: str) -> str:
    return path + (f"?{query}" if query else "") + (f"#{fragment}" if fragment else "")


def rewrite_url(value: str, page_dir: str, site_url: str) -> str:
    """Root path (or own absolute URL) with clean page paths, for a URL found in a page."""
    origin, base_path = split_site_url(site_url)
    if value == "" or value.startswith("#"):
        return value
    if value.startswith(origin + "/"):
        parts = urlsplit(value)
        return origin + _join(clean_path(parts.path, base_path), parts.query, parts.fragment)
    if SCHEME.match(value) or value.startswith("//"):
        return value
    parts = urlsplit(urljoin(page_dir, value))
    return _join(clean_path(parts.path, base_path), parts.query, parts.fragment)


def search_location(location: str, site_url: str) -> str:
    """Search result location that the theme resolves to a clean URL.

    The theme opens a result at `new URL("./" + location, base + "/")`: "guides/x/"
    becomes "guides/x", and the home page ("" or "#anchor") "../docs" (the base's
    own last segment), since "./" alone would keep the trailing slash.
    """
    _, base_path = split_site_url(site_url)
    parts = urlsplit(urljoin(base_path + "/", location))
    path = clean_path(parts.path, base_path)
    relative = (
        "../" + base_path.rsplit("/", 1)[-1] if path == base_path else path[len(base_path) + 1 :]
    )
    return _join(relative, parts.query, parts.fragment)


def page_url(rel_file: str, site_url: str) -> str:
    """Clean URL a page file is served at: "guides/x/index.html" -> <site_url>/guides/x."""
    origin, base_path = split_site_url(site_url)
    return origin + clean_path(f"{base_path}/{rel_file}", base_path)


def rewrite_html(text: str, page_dir: str, site_url: str) -> str:
    def url(match: re.Match) -> str:
        return match.group(1) + rewrite_url(match.group(2), page_dir, site_url) + match.group(3)

    def scope(match: re.Match) -> str:
        # A directory: the trailing slash stays (the theme adds one anyway).
        directory = match.group(2).rstrip("/") + "/"
        return match.group(1) + urlsplit(urljoin(page_dir, directory)).path + match.group(3)

    def config(match: re.Match) -> str:
        data = json.loads(match.group(2))
        for key in ("base", "search"):
            if isinstance(data.get(key), str):
                data[key] = rewrite_url(data[key], page_dir, site_url)
        return (
            match.group(1)
            + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            + match.group(3)
        )

    text = URL_ATTRIBUTE.sub(url, text)
    text = SCOPE_SCRIPT.sub(scope, text)
    text = CONFIG_SCRIPT.sub(config, text)
    return text.replace(SOURCE_FACTS, "")


def rewrite_site(site_dir: Path, site_url: str) -> None:
    """Rewrite a built documentation in place for its clean URLs (idempotent)."""
    _, base_path = split_site_url(site_url)
    # The website's distribution answers unknown paths with its own 404 page and a
    # 404 status; the theme's page would only be one more indexable URL.
    (site_dir / "404.html").unlink(missing_ok=True)
    for path in site_dir.rglob("*.html"):
        rel = path.relative_to(site_dir).as_posix()
        page_dir = f"{base_path}/{rel}".rsplit("/", 1)[0] + "/"
        text = path.read_text(encoding="utf-8")
        path.write_text(rewrite_html(text, page_dir, site_url), encoding="utf-8")
    index = site_dir / "search.json"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            item["location"] = search_location(item["location"], site_url)
        index.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
    sitemap = site_dir / "sitemap.xml"
    if sitemap.exists():
        text = sitemap.read_text(encoding="utf-8")
        text = SITEMAP_LOC.sub(
            lambda m: m.group(1) + rewrite_url(m.group(2), "/", site_url) + m.group(3), text
        )
        sitemap.write_text(text, encoding="utf-8")


# --- Check ---------------------------------------------------------------------


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self.canonicals: list[str] = []
        self.config: str | None = None
        self._in_config = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        for name in ("href", "src"):
            if attributes.get(name) is not None:
                self.urls.append(attributes[name])
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonicals.append(attributes.get("href") or "")
        self._in_config = tag == "script" and attributes.get("id") == "__config"

    def handle_data(self, data):
        if self._in_config:
            self.config = data
            self._in_config = False


def _target_exists(site_dir: Path, path: str, base_path: str) -> bool:
    rel = path[len(base_path) :].lstrip("/")
    last = rel.rsplit("/", 1)[-1]
    if "." in last:
        return (site_dir / rel).is_file()
    return (site_dir / rel / "index.html").is_file()


def _check_path(site_dir: Path, where: str, path: str, base_path: str) -> list[str]:
    if path != base_path and not path.startswith(base_path + "/"):
        return []
    if path.endswith("/"):
        return [f"{where}: {path} has a trailing slash (the CDN answers it with a 301)"]
    if path.endswith("/index.html"):
        return [f"{where}: {path} names index.html instead of the clean URL"]
    if not _target_exists(site_dir, path, base_path):
        return [f"{where}: {path} points to nothing in the build"]
    return []


def check_site(site_dir: Path, site_url: str) -> list[str]:
    """Problems that would break the published documentation (empty list: none)."""
    origin, base_path = split_site_url(site_url)
    problems: list[str] = []
    if (site_dir / "404.html").exists():
        problems.append("404.html: the website's 404 page serves unknown paths, drop this one")
    pages = sorted(site_dir.rglob("*.html"))
    if not pages:
        problems.append(f"{site_dir}: no page")
    for page in pages:
        rel = page.relative_to(site_dir).as_posix()
        text = page.read_text(encoding="utf-8")
        for needle in FORBIDDEN:
            if needle in text:
                problems.append(f"{rel}: references {needle}")
        parser = _Links()
        parser.feed(text)
        if parser.canonicals != [page_url(rel, site_url)]:
            problems.append(
                f"{rel}: canonical {parser.canonicals} instead of {page_url(rel, site_url)}"
            )
        for value in parser.urls:
            if value == "" or value.startswith("#"):
                continue
            if value.startswith(origin + "/"):
                problems += _check_path(site_dir, rel, urlsplit(value).path, base_path)
            elif SCHEME.match(value) or value.startswith("//"):
                continue
            elif not value.startswith("/"):
                problems.append(
                    f"{rel}: relative URL {value} (resolved one level too high without a trailing slash)"
                )
            else:
                problems += _check_path(site_dir, rel, urlsplit(value).path, base_path)
        if parser.config is None:
            problems.append(f"{rel}: no theme configuration (__config)")
        else:
            config = json.loads(html.unescape(parser.config))
            if config.get("base") != base_path:
                problems.append(
                    f"{rel}: __config base {config.get('base')!r} instead of {base_path!r}"
                )
            problems += _check_path(
                site_dir, f"{rel} (search worker)", config.get("search", ""), base_path
            )
    index = site_dir / "search.json"
    if not index.is_file():
        problems.append("search.json: missing, the search would find nothing")
    else:
        for item in json.loads(index.read_text(encoding="utf-8")).get("items", []):
            # Same resolution as the theme: new URL("./" + location, base + "/").
            path = urlsplit(urljoin(base_path + "/", "./" + item["location"])).path
            problems += _check_path(
                site_dir, f"search.json ({item['location']!r})", path, base_path
            )
    sitemap = site_dir / "sitemap.xml"
    if not sitemap.is_file():
        problems.append("sitemap.xml: missing")
    else:
        for loc in SITEMAP_LOC.findall(sitemap.read_text(encoding="utf-8")):
            url = loc[1]
            if not url.startswith(site_url):
                problems.append(f"sitemap.xml: {url} is outside {site_url}")
            else:
                problems += _check_path(site_dir, "sitemap.xml", urlsplit(url).path, base_path)
    return problems


# --- Redirects of the former host ------------------------------------------------

TITLE = re.compile(r"<title>(.*?)</title>", re.S)


def redirect_page(title: str, target: str) -> str:
    """A page that sends the reader (and search engines) to `target`.

    GitHub Pages cannot answer a 301. Search engines treat an immediate meta
    refresh as a permanent redirect and the canonical names the new URL; the script
    runs first so that the fragment (#section) survives, which the refresh drops.
    """
    t = html.escape(target, quote=True)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f'<link rel="canonical" href="{t}">\n'
        f"<script>location.replace({json.dumps(target)} + location.hash)</script>\n"
        f'<meta http-equiv="refresh" content="0; url={t}">\n'
        "</head>\n<body>\n"
        f'<p>This page has moved to <a href="{t}">{t}</a>.</p>\n'
        "</body>\n</html>\n"
    )


def not_found_page(site_url: str, old_base_path: str) -> str:
    """404 page of the former host: any former path goes to the same page on the new one."""
    target = html.escape(site_url, quote=True)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>Erudi Documentation has moved</title>\n"
        "<script>\n"
        f"var base = {json.dumps(old_base_path)}, target = {json.dumps(site_url)};\n"
        "var path = location.pathname;\n"
        'path = path === base || path.indexOf(base + "/") === 0 ? path.slice(base.length) : "";\n'
        'path = path.replace(/\\/index\\.html$/, "").replace(/\\/+$/, "");\n'
        "location.replace(target + path + location.hash);\n"
        "</script>\n"
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        "</head>\n<body>\n"
        f'<p>Erudi Documentation has moved to <a href="{target}">{target}</a>.</p>\n'
        "</body>\n</html>\n"
    )


def write_redirects(site_dir: Path, out_dir: Path, site_url: str, old_base_path: str) -> None:
    """One redirect page per page of the documentation, at its former address."""
    for page in sorted(site_dir.rglob("index.html")):
        rel = page.relative_to(site_dir).as_posix()
        match = TITLE.search(page.read_text(encoding="utf-8"))
        title = match.group(1).strip() if match else "Erudi Documentation"
        stub = out_dir / rel
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(redirect_page(title, page_url(rel, site_url)), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "404.html").write_text(not_found_page(site_url, old_base_path), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rewrite", "check", "redirects"):
        command = sub.add_parser(name)
        command.add_argument("site", type=Path)
        if name == "redirects":
            command.add_argument("out", type=Path)
            command.add_argument("--old-base-path", required=True)
        command.add_argument("--site-url", required=True)
    args = parser.parse_args(argv)
    site_url = args.site_url.rstrip("/")
    if args.command == "rewrite":
        rewrite_site(args.site, site_url)
    elif args.command == "redirects":
        write_redirects(args.site, args.out, site_url, args.old_base_path.rstrip("/"))
    else:
        problems = check_site(args.site, site_url)
        for problem in problems:
            print(f"::error title=Documentation::{problem}")
        if problems:
            return 1
        print(f"{args.site}: {len(list(args.site.rglob('*.html')))} pages ready for {site_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
