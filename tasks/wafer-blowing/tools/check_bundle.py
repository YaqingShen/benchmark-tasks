"""Verify the complete task runtime without starting Isaac."""
import hashlib
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
RUNTIME = TASK / 'runtime'


def check():
    manifest = json.loads((RUNTIME / 'bundle_files.json').read_text())
    if len(manifest) != 47:
        raise ValueError('Expected the complete 47-file particle-policy runtime.')
    for name, expected in manifest.items():
        path = (RUNTIME / name).resolve()
        if not path.is_relative_to(RUNTIME.resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe manifest path: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source checksum mismatch: ' + name)
    return dict(files_verified=len(manifest), checksum_mismatches=0)


if __name__ == '__main__':
    print(json.dumps(check()))
