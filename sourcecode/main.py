"""Run analysis with the prepared local environment; no installation or ZIP creation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True


def use_project_environment(arguments):
    """Use the existing .venv without installing packages or changing global Python."""
    if os.name == 'nt':
        parts = [part for part in str(ROOT).replace('\\', '/').split('/') if part]
        if parts and parts[0].lower() in ['wsl.localhost', 'wsl$']:
            distro, linux_root = parts[1], '/' + '/'.join(parts[2:])
            arguments = arguments.copy()
            for index, argument in enumerate(arguments):
                if index and arguments[index - 1] == '--output-dir':
                    arguments[index] = subprocess.check_output(
                        ['wsl.exe', '-d', distro, 'wslpath', '-a', argument], text=True).strip()
            raise SystemExit(subprocess.call(
                ['wsl.exe', '-d', distro, '--cd', linux_root,
                 linux_root + '/.venv/bin/python', linux_root + '/main.py', *arguments]))
    environment = ROOT / '.venv'
    interpreter = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if Path(sys.prefix).resolve() == environment.resolve():
        return
    if interpreter.exists():
        raise SystemExit(subprocess.call([str(interpreter), str(ROOT / 'main.py'), *arguments]))
    # A recipient may explicitly supply another already prepared environment.
    try:
        import numpy
        import pandas
        import sklearn
        import matplotlib
    except ImportError:
        raise SystemExit('Prepared environment not found. Create .venv with Python 3.12 and install requirements.txt.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'forecast', 'segments'], default='all')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent,
                        help='Submission folder (default: parent of sourcecode).')
    args = parser.parse_args()
    use_project_environment(sys.argv[1:])
    sys.path.insert(0, str(ROOT / 'src'))
    from pipeline import run
    run(ROOT, args.output_dir.resolve(), args.stage)


if __name__ == '__main__':
    main()
