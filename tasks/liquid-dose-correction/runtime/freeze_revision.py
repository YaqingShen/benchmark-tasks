"""Create an immutable transport archive of this task's source and assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('revision');a=p.parse_args()
    if not re.fullmatch(r'revision\d{3}',a.revision):raise SystemExit('Expected revisionNNN')
    root=Path(__file__).resolve().parent
    out=root.parent.parent/'outputs/formulation-prototype'/f'formulation-native-{a.revision}.tar.gz'
    if out.exists():raise SystemExit('Do not replace a frozen revision')
    files=[f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts
           and f.name!='source_manifest.json' and f.suffix in ('.py','.json','.md','.usda')]
    manifest={'revision':a.revision,'files':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}}
    mp=root/'source_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(out,'w:gz') as archive:
        for f in files+[mp]:archive.add(f,arcname=str(f.relative_to(root)))
    print(json.dumps({'path':str(out),'files':len(files),'bytes':out.stat().st_size,
                      'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
