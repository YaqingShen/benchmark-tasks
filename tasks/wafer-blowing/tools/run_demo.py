"""Launch the frozen native particle demo in the active Isaac environment."""
import argparse
import subprocess
import sys
from pathlib import Path

from check_bundle import RUNTIME, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--show-command', action='store_true')
    args = parser.parse_args()
    check()
    destination = args.output.resolve()
    if destination.exists():
        parser.error('Choose a new output directory; existing evidence is preserved.')
    command = [sys.executable, '-u', str(RUNTIME / 'wafer_dust_isaac.py'),
               '--bundle', str(RUNTIME), '--output', str(destination),
               '--device', 'cpu', '--development']
    if args.show_command:
        import shlex
        print(shlex.join(command))
        return
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
