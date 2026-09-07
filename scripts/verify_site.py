"""Check built documentation routes, assets, access boundary and source links."""
from pathlib import Path
import argparse
import json
import re
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {
    'political-risk-llm-nowcasting',
    'political-risk-established-baselines',
    'political-risk-dissertation-guide',
}
SITE = 'political-risk-dissertation-guide.rahulnundlall.chatgpt.site'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-parent', type=Path, help='Also resolve all three public GitHub source routes against sibling candidate checkouts.')
    args = parser.parse_args()
    build = ROOT / 'dist'
    pages = {p: BeautifulSoup(p.read_text(), 'html.parser') for p in build.rglob('*.html')}
    assert len(pages) == 23, f'Expected 22 reading pages and 404, found {len(pages)}'
    links = 0
    github = set()
    for path, soup in pages.items():
        assert 'Documentation prepared with AI assistance' in soup.get_text(), f'Missing footer: {path}'
        for node in soup.find_all(True):
            for attribute in ('href', 'src'):
                value = node.get(attribute)
                if not value or value.startswith(('mailto:', 'data:', 'javascript:')):
                    continue
                parts = urlsplit(value)
                if parts.netloc == 'github.com' and parts.path.startswith('/skyboyrahul/'):
                    tokens = unquote(parts.path).strip('/').split('/')
                    assert tokens[1] in PUBLIC, f'Private repository link: {value}'
                    github.add(value)
                    if args.repo_parent:
                        local = args.repo_parent / tokens[1]
                        if len(tokens) > 2:
                            assert len(tokens) >= 4 and tokens[2] in {'blob', 'tree'} and tokens[3] == 'main', value
                            local = local.joinpath(*tokens[4:])
                        assert local.exists(), f'Missing public source destination: {value}'
                    continue
                if parts.scheme and parts.netloc != SITE:
                    continue
                if parts.netloc and parts.netloc != SITE:
                    continue
                route = unquote(parts.path)
                if not route:
                    target = path
                elif route.startswith('/'):
                    target = build / route.lstrip('/')
                else:
                    target = path.parent / route
                target = target.resolve()
                assert target.is_relative_to(build.resolve()), f'Outside build: {value}'
                if target.is_dir():
                    target /= 'index.html'
                assert target.is_file(), f'Broken local route or asset: {path.relative_to(build)} -> {value}'
                if parts.fragment and target.suffix == '.html':
                    target_soup = pages.get(target) or BeautifulSoup(target.read_text(), 'html.parser')
                    anchor = unquote(parts.fragment)
                    assert target_soup.find(id=anchor) or target_soup.find(attrs={'name': anchor}), f'Missing anchor: {value}'
                links += 1
    assert len(github) >= 20, 'Missing expected source links'
    for p in build.rglob('*'):
        if p.is_file():
            assert p.suffix.lower() not in {'.csv', '.tsv', '.zip', '.pdf', '.parquet', '.py'}, f'Research input/source in site assets: {p}'
            if p.suffix in {'.html', '.json', '.js', '.css', '.txt'}:
                data = p.read_text(errors='replace')
                for forbidden in ('/Users/', 'students.wits.ac.za'):
                    assert forbidden not in data, f'Private reference in built asset: {p.relative_to(build)}'
                for repo in re.findall(r'github\.com/skyboyrahul/([a-zA-Z0-9_-]+)', data):
                    assert repo in PUBLIC, f'Non-public research repository in built asset: {p.relative_to(build)}'
    homepage = pages[build / 'index.html']
    assert homepage.find(id='ai-assistance'), 'Missing AI-assistance disclosure section'
    assert 'generated and edited with the assistance of generative AI' in homepage.get_text()
    assert sum(len(s.select('.mermaid')) for s in pages.values()) >= 3, 'Missing research diagrams'
    print(json.dumps({'html_pages': len(pages), 'local_links_and_assets': links, 'public_github_links': len(github), 'source_destinations_checked': bool(args.repo_parent), 'restricted_asset_extensions': 0}, indent=2))


if __name__ == '__main__':
    main()
