"""Verify the reviewed public file inventory; run from any checkout directory."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'public-files.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-untracked', action='store_true', help='Also check non-ignored candidate files before the initial commit.')
    args = parser.parse_args()
    inventory = json.loads((ROOT / MANIFEST).read_text())
    assert inventory['policy'] == 'explicit-reviewed-public-files-v1'
    expected = {row['path']: row for row in inventory['files']}
    assert len(expected) == len(inventory['files']), 'Duplicate inventory path'
    assert MANIFEST not in expected, 'Inventory must not hash itself'
    command = ['git', 'ls-files', '--cached', '-z']
    if args.include_untracked:
        command += ['--others', '--exclude-standard']
    observed = set(subprocess.check_output(command, cwd=ROOT).decode().split('\0')) - {''}
    assert observed == set(expected) | {MANIFEST}, (
        'Publication inventory mismatch. Missing: ' + str(sorted((set(expected) | {MANIFEST}) - observed))
        + '; unreviewed: ' + str(sorted(observed - set(expected) - {MANIFEST}))
    )
    for relative, record in expected.items():
        path = ROOT / relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert not path.is_symlink() and path.is_file(), f'Non-regular file: {relative}'
        data = path.read_bytes()
        assert len(data) == record['bytes'], f'Size changed; review required: {relative}'
        assert hashlib.sha256(data).hexdigest() == record['sha256'], f'Content changed; review required: {relative}'
        assert path.suffix.lower() not in {'.csv', '.tsv', '.parquet', '.pkl', '.pickle', '.npy', '.npz', '.zip', '.gz', '.pdf', '.png', '.jpg', '.jpeg', '.ipynb'}, f'Unexpected data or binary asset: {relative}'
    print(f'Public inventory verified: {len(expected)} reviewed files plus the inventory; no unreviewed tracked files.')
    print('Hashes detect changes to the reviewed release. Updating the inventory requires another content review; this is not a licence or empirical-reproduction check.')


if __name__ == '__main__':
    main()
