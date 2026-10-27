"""Tests of prepare_site.py (stdlib only, no documentation tooling needed).

Run from the repository root:
    python -m unittest discover -s scripts/docs -p "test_*.py"
"""

import json
import tempfile
import unittest
from pathlib import Path

import prepare_site as ps

SITE_URL = "https://www.erudi.app/open-source/desktop/docs"
BASE = "/open-source/desktop/docs"


def page(body: str, *, canonical: str, base: str = "../..", scope: str = "../..") -> str:
    """A page shaped like the theme's output: relative URLs, trailing slashes."""
    config = {
        "base": base,
        "search": f"{base}/assets/javascripts/workers/search.1234abcd.min.js",
        "features": [],
    }
    return (
        "<!doctype html><html><head>"
        f'<link rel="canonical" href="{canonical}">'
        f'<link rel="stylesheet" href="{base}/assets/stylesheets/main.1234abcd.min.css">'
        f'<script>__md_scope=new URL("{scope}",location),__md_get=1</script>'
        "<title>Hardware detection - Erudi Documentation</title>"
        "</head><body>"
        '<a href="https://github.com/erudi-app/erudi" title="Go to repository" '
        'class="md-source" data-md-component="source">repo</a>'
        f"{body}"
        f'<script id="__config" type="application/json">{json.dumps(config)}</script>'
        f'<script src="{base}/assets/javascripts/bundle.1234abcd.min.js"></script>'
        "</body></html>"
    )


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fake_site(root: Path) -> None:
    """A miniature build: home, two nested pages, assets, search index, sitemap, 404."""
    write(
        root,
        "index.html",
        page(
            '<a href="guides/hardware/">Hardware</a><a href="./">Home</a>'
            '<a href="#overview">Overview</a><img src="assets/screenshots/download.png">',
            canonical=f"{SITE_URL}/",
            base=".",
            scope=".",
        ),
    )
    write(
        root,
        "guides/hardware/index.html",
        page(
            '<a href="../llms/#sizes">Models</a><a href="../..">Home</a>'
            '<a href="../../reference/core/index.html#src.core">Core</a>'
            '<a href="">Self</a><a href="#detection">Here</a>',
            canonical=f"{SITE_URL}/guides/hardware/",
        ),
    )
    write(root, "guides/llms/index.html", page("", canonical=f"{SITE_URL}/guides/llms/"))
    write(root, "reference/core/index.html", page("", canonical=f"{SITE_URL}/reference/core/"))
    write(root, "404.html", page(f'<a href="{BASE}/">Home</a>', canonical=f"{SITE_URL}/"))
    for asset in (
        "assets/stylesheets/main.1234abcd.min.css",
        "assets/javascripts/bundle.1234abcd.min.js",
        "assets/javascripts/workers/search.1234abcd.min.js",
        "assets/screenshots/download.png",
    ):
        write(root, asset, "x")
    write(
        root,
        "search.json",
        json.dumps(
            {
                "config": {},
                "items": [
                    {"location": "", "title": "Home"},
                    {"location": "#overview", "title": "Overview"},
                    {"location": "guides/hardware/", "title": "Hardware"},
                    {"location": "guides/hardware/#detection", "title": "Detection"},
                ],
            }
        ),
    )
    write(
        root,
        "sitemap.xml",
        "<urlset>"
        f"<url><loc>{SITE_URL}/</loc></url>"
        f"<url><loc>{SITE_URL}/guides/hardware/</loc></url>"
        "</urlset>",
    )


class CleanPathTest(unittest.TestCase):
    def test_page_paths_lose_trailing_slash_and_index(self):
        self.assertEqual(ps.clean_path(f"{BASE}/guides/hardware/", BASE), f"{BASE}/guides/hardware")
        self.assertEqual(ps.clean_path(f"{BASE}/a/index.html", BASE), f"{BASE}/a")
        self.assertEqual(ps.clean_path(f"{BASE}/", BASE), BASE)
        self.assertEqual(ps.clean_path(f"{BASE}/index.html", BASE), BASE)

    def test_files_and_foreign_paths_are_untouched(self):
        self.assertEqual(ps.clean_path(f"{BASE}/search.json", BASE), f"{BASE}/search.json")
        self.assertEqual(ps.clean_path("/open-source/desktop/", BASE), "/open-source/desktop/")
        self.assertEqual(ps.clean_path(f"{BASE}x/", BASE), f"{BASE}x/")


class RewriteUrlTest(unittest.TestCase):
    page_dir = f"{BASE}/guides/hardware/"

    def rewrite(self, value):
        return ps.rewrite_url(value, self.page_dir, SITE_URL)

    def test_relative_page_link_becomes_clean_root_path(self):
        self.assertEqual(self.rewrite("../llms/#sizes"), f"{BASE}/guides/llms#sizes")
        self.assertEqual(self.rewrite("../.."), BASE)
        self.assertEqual(
            self.rewrite("../../reference/core/index.html#x"), f"{BASE}/reference/core#x"
        )

    def test_relative_asset_becomes_root_path(self):
        self.assertEqual(
            self.rewrite("../../assets/javascripts/bundle.1234abcd.min.js"),
            f"{BASE}/assets/javascripts/bundle.1234abcd.min.js",
        )

    def test_fragments_empty_and_foreign_urls_are_untouched(self):
        for value in ("", "#detection", "https://github.com/erudi-app/erudi", "mailto:a@b.c"):
            self.assertEqual(self.rewrite(value), value)

    def test_absolute_url_of_the_docs_is_cleaned(self):
        self.assertEqual(
            self.rewrite(f"{SITE_URL}/guides/hardware/"), f"{SITE_URL}/guides/hardware"
        )
        self.assertEqual(self.rewrite(f"{SITE_URL}/"), SITE_URL)

    def test_root_path_with_trailing_slash_is_cleaned(self):
        self.assertEqual(self.rewrite(f"{BASE}/"), BASE)

    def test_query_is_kept(self):
        self.assertEqual(self.rewrite("../llms/?q=a#b"), f"{BASE}/guides/llms?q=a#b")


class SearchLocationTest(unittest.TestCase):
    """The theme opens a result at new URL("./" + location, base + "/")."""

    def test_locations_resolve_to_clean_urls(self):
        cases = {
            "": "../docs",
            "#overview": "../docs#overview",
            "guides/hardware/": "guides/hardware",
            "guides/hardware/#detection": "guides/hardware#detection",
        }
        for location, expected in cases.items():
            self.assertEqual(ps.search_location(location, SITE_URL), expected, location)


class RewriteSiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fake_site(self.root)
        ps.rewrite_site(self.root, SITE_URL)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, rel):
        return (self.root / rel).read_text(encoding="utf-8")

    def test_nested_page(self):
        html = self.read("guides/hardware/index.html")
        self.assertIn(f'<link rel="canonical" href="{SITE_URL}/guides/hardware">', html)
        self.assertIn(f'href="{BASE}/guides/llms#sizes"', html)
        self.assertIn(f'<a href="{BASE}">Home</a>', html)
        self.assertIn(f'href="{BASE}/reference/core#src.core"', html)
        self.assertIn('<a href="">Self</a><a href="#detection">Here</a>', html)
        self.assertIn(f'href="{BASE}/assets/stylesheets/main.1234abcd.min.css"', html)
        self.assertIn(f'new URL("{BASE}/",location)', html)
        self.assertNotIn('data-md-component="source"', html)
        self.assertIn('class="md-source"', html)
        config = json.loads(html.split('type="application/json">')[1].split("</script>")[0])
        self.assertEqual(config["base"], BASE)
        self.assertEqual(
            config["search"], f"{BASE}/assets/javascripts/workers/search.1234abcd.min.js"
        )

    def test_home_page(self):
        html = self.read("index.html")
        self.assertIn(f'<link rel="canonical" href="{SITE_URL}">', html)
        self.assertIn(f'href="{BASE}/guides/hardware"', html)
        self.assertIn(f'src="{BASE}/assets/screenshots/download.png"', html)
        self.assertIn(f'<a href="{BASE}">Home</a>', html)

    def test_search_index_and_sitemap(self):
        items = json.loads(self.read("search.json"))["items"]
        self.assertEqual(
            [i["location"] for i in items],
            ["../docs", "../docs#overview", "guides/hardware", "guides/hardware#detection"],
        )
        sitemap = self.read("sitemap.xml")
        self.assertIn(f"<loc>{SITE_URL}</loc>", sitemap)
        self.assertIn(f"<loc>{SITE_URL}/guides/hardware</loc>", sitemap)

    def test_theme_404_page_is_dropped(self):
        # The site's distribution serves its own 404 page, with a 404 status.
        self.assertFalse((self.root / "404.html").exists())

    def test_rewritten_site_passes_the_check(self):
        self.assertEqual(ps.check_site(self.root, SITE_URL), [])

    def test_rewrite_is_idempotent(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        ps.rewrite_site(self.root, SITE_URL)
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


class CheckSiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fake_site(self.root)
        ps.rewrite_site(self.root, SITE_URL)

    def tearDown(self):
        self.tmp.cleanup()

    def add_link(self, rel, link):
        path = self.root / rel
        path.write_text(
            path.read_text(encoding="utf-8").replace("</body>", f"{link}</body>"), encoding="utf-8"
        )

    def assertProblem(self, fragment):
        problems = ps.check_site(self.root, SITE_URL)
        self.assertTrue(any(fragment in p for p in problems), problems)

    def test_unbuilt_site_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            fake_site(Path(raw))
            problems = ps.check_site(Path(raw), SITE_URL)
        self.assertTrue(any("relative URL" in p for p in problems), problems)
        self.assertTrue(any("canonical" in p for p in problems), problems)

    def test_broken_internal_link(self):
        self.add_link("index.html", f'<a href="{BASE}/nowhere">x</a>')
        self.assertProblem(f"{BASE}/nowhere")

    def test_missing_asset(self):
        self.add_link("index.html", f'<img src="{BASE}/assets/missing.png">')
        self.assertProblem(f"{BASE}/assets/missing.png")

    def test_trailing_slash_link(self):
        self.add_link("index.html", f'<a href="{BASE}/guides/llms/">x</a>')
        self.assertProblem("trailing slash")

    def test_third_party_font_and_repository_facts(self):
        self.add_link("index.html", '<link href="https://fonts.googleapis.com/css?family=Inter">')
        self.assertProblem("fonts.googleapis.com")
        self.add_link("index.html", '<a data-md-component="source">x</a>')
        self.assertProblem('data-md-component="source"')

    def test_former_host(self):
        self.add_link("index.html", '<a href="https://erudi-app.github.io/erudi/privacy/">x</a>')
        self.assertProblem("erudi-app.github.io")

    def test_missing_search_index(self):
        (self.root / "search.json").unlink()
        self.assertProblem("search.json")


class RedirectsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.site = Path(self.tmp.name) / "site"
        self.out = Path(self.tmp.name) / "redirects"
        fake_site(self.site)
        ps.rewrite_site(self.site, SITE_URL)
        ps.write_redirects(self.site, self.out, SITE_URL, "/erudi")

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_stub_per_page_at_the_former_address(self):
        stubs = sorted(p.relative_to(self.out).as_posix() for p in self.out.rglob("index.html"))
        self.assertEqual(
            stubs,
            [
                "guides/hardware/index.html",
                "guides/llms/index.html",
                "index.html",
                "reference/core/index.html",
            ],
        )

    def test_stub_redirects_keeps_the_fragment_and_declares_the_canonical(self):
        html = (self.out / "guides/hardware/index.html").read_text(encoding="utf-8")
        target = f"{SITE_URL}/guides/hardware"
        self.assertIn(f'<link rel="canonical" href="{target}">', html)
        self.assertIn(f'<meta http-equiv="refresh" content="0; url={target}">', html)
        self.assertIn(f'location.replace("{target}" + location.hash)', html)
        self.assertIn(f'<a href="{target}">', html)
        self.assertIn("<title>Hardware detection - Erudi Documentation</title>", html)
        # The script runs before the refresh, so the fragment is kept.
        self.assertLess(html.index("location.replace"), html.index('http-equiv="refresh"'))

    def test_home_stub(self):
        html = (self.out / "index.html").read_text(encoding="utf-8")
        self.assertIn(f'content="0; url={SITE_URL}"', html)

    def test_not_found_page_maps_any_former_path(self):
        html = (self.out / "404.html").read_text(encoding="utf-8")
        self.assertIn('"/erudi"', html)
        self.assertIn(f'"{SITE_URL}"', html)
        self.assertIn(f'content="0; url={SITE_URL}"', html)

    def test_only_stubs_are_published(self):
        files = sorted(
            p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()
        )
        self.assertTrue(all(f.endswith(".html") for f in files), files)


if __name__ == "__main__":
    unittest.main()
