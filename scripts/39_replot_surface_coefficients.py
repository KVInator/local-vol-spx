#!/usr/bin/env python3
"""Correct coefficient figures from pinned saved samples, without model evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import surface_evidence
from surface_evidence import plot_coefficient_evidence, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-run', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    clock = time.perf_counter()
    run = args.evidence_run.resolve()
    audit_path = run / 'audit.json'
    audit_hash = sha256(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'completed':
        raise ValueError('Select a completed surface-evidence run.')
    pins = {str(audit_path): audit_hash}
    sources = {str(Path(__file__).resolve()): sha256(__file__),
               str(Path(surface_evidence.__file__).resolve()): sha256(surface_evidence.__file__)}

    def checked(relative):
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or relative not in audit['output_sha256']:
            raise ValueError(f'Unpinned or escaping saved-output path: {relative}')
        expected = audit['output_sha256'][relative]
        if sha256(path) != expected:
            raise ValueError(f'Saved-output digest mismatch: {relative}')
        pins[str(path)] = expected
        return path

    status = pd.read_csv(checked('evidence_status.csv'))
    if not status.status.eq('completed').all():
        raise ValueError('All requested evidence identities must be completed.')
    samples = [(row, checked(row.samples_file)) for row in status.itertuples(index=False)]
    parent = (args.output or run / 'coefficient_plot_corrections').resolve()
    if parent == run or run.is_relative_to(parent):
        raise ValueError('Correction output must be a separate subfolder or unrelated folder.')
    folder = parent / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for row, path in samples:
        frame = pd.read_csv(path)
        prefix = Path(str(row.plot_prefix)).name
        plot_coefficient_evidence(frame, folder / prefix)
        records.append({'quote_date': row.quote_date, 'root': row.root,
                        'figure': prefix + '_coefficients.png',
                        'input_samples_file': str(path)})
    for path, expected in {**pins, **sources}.items():
        if sha256(path) != expected:
            raise ValueError(f'Input or source changed during plotting: {path}')
    output_hashes = {item['figure']: sha256(folder / item['figure']) for item in records}
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Corrected early-coefficient figures\n',
            'One positive-time slice, one derivative side, increasing log-state.\n',
            'Saved numerical values are unchanged. Set ROOT if Jupyter uses another working directory.']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
            'from pathlib import Path\nimport hashlib, json\nfrom IPython.display import Image, display\n',
            'ROOT = Path.cwd()\n',
            "audit = json.loads((ROOT / 'audit.json').read_text())\n",
            "for item in audit['figures']:\n",
            "    path = ROOT / item['figure']\n",
            "    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit['output_sha256'][item['figure']]\n",
            "    print(item['quote_date'], item['root'])\n",
            "    display(Image(filename=str(path)))\n"]}]}
    notebook_path = folder / 'coefficients_corrected.ipynb'
    notebook_path.write_text(json.dumps(notebook, indent=2) + '\n')
    output_hashes[notebook_path.name] = sha256(notebook_path)
    result = dict(status='completed', original_evidence_run=str(run), figures=records,
                  fix='Earliest positive time, explicit left/right side, increasing y.',
                  original_outputs_modified=False, numerical_values_changed=False,
                  models_loaded=False, surface_evaluated=False, pde_executed=False,
                  input_sha256=pins, source_sha256=sources, output_sha256=output_hashes,
                  elapsed_wall_seconds=time.perf_counter()-clock)
    (folder / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Corrected {len(records)} coefficient figures from verified saved samples.')
    print(f'Measured wall time: {result["elapsed_wall_seconds"]:.3f} seconds.')
    print(f'Notebook: {notebook_path}')
    print('Original evidence files and numerical results are unchanged.')


if __name__ == '__main__':
    main()
