import os
import json
import argparse
from pathlib import Path

try:
    import papermill as pm
except ImportError:
    pm = None


def run_notebook(input_nb, output_nb=None, params=None, kernel_name='python3', timeout=600):
    input_path = Path(input_nb)
    if not input_path.exists():
        raise FileNotFoundError(f'Input notebook not found: {input_path}')

    if output_nb is None:
        output_path = input_path.with_name(input_path.stem + '_executed' + input_path.suffix)
    else:
        output_path = Path(output_nb)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pm is None:
        raise ImportError('papermill is not installed. Install via pip install papermill')

    print(f'Running notebook: {input_path} -> {output_path}')
    pm.execute_notebook(
        input_path=str(input_path),
        output_path=str(output_path),
        parameters=params or {},
        kernel_name=kernel_name,
        progress_bar=True,
        report_mode=False,
        start_timeout=timeout,
        execution_timeout=timeout,
    )

    print('Done:', output_path)
    return str(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a notebook via papermill')
    parser.add_argument('--input', '-i', required=False, default=None, help='Input notebook path (optional)')
    parser.add_argument('--output', '-o', default=None, help='Output notebook path')
    parser.add_argument('--all', action='store_true', help='Run all notebooks in the project (notebooks folder)')
    parser.add_argument('--params', '-p', default=None, help='JSON string of parameters')
    parser.add_argument('--params-file', default=None, help='JSON file with parameters')
    parser.add_argument('--kernel', default='python3', help='Kernel name')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds')
    args = parser.parse_args()

    if args.all:
        if args.input:
            raise ValueError('Cannot use --all and --input together')

        candidate_dirs = [Path.cwd(), Path.cwd().parent / 'notebooks', Path.cwd() / 'notebooks']
        notebooks = []
        for d in candidate_dirs:
            if d.exists() and d.is_dir():
                notebooks.extend(sorted(d.glob('*.ipynb')))

        if not notebooks:
            raise FileNotFoundError('No .ipynb notebooks found in current or notebooks folders')

        for nb in notebooks:
            print(f'Running notebook: {nb}')
            run_notebook(str(nb), args.output, params=params, kernel_name=args.kernel, timeout=args.timeout)
        print(f'Completed all {len(notebooks)} notebook(s).')
        exit(0)

    if args.input is None:
        # try to auto-detect a notebook in common locations
        candidate_dirs = [Path.cwd(), Path.cwd().parent / 'notebooks', Path.cwd() / 'notebooks']
        found = None
        for d in candidate_dirs:
            if d.exists() and d.is_dir():
                notebooks = sorted(d.glob('*.ipynb'))
                if notebooks:
                    found = notebooks[0]
                    break
        if found is None:
            raise FileNotFoundError('No notebook input provided and no .ipynb found in current/../notebooks')

        print(f"No --input provided. Auto-selected notebook: {found}")
        args.input = str(found)

    params = None
    if args.params:
        params = json.loads(args.params)
    elif args.params_file:
        params_path = Path(args.params_file)
        if not params_path.exists():
            raise FileNotFoundError(f'Params file not found: {params_path}')
        params = json.loads(params_path.read_text(encoding='utf-8'))

    run_notebook(args.input, args.output, params=params, kernel_name=args.kernel, timeout=args.timeout)

