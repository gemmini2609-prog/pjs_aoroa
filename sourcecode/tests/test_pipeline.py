"""Verify standalone CLI execution and the unchanged submission contract."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_default_cli_reproduces_both_submissions_from_other_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results'
            output.mkdir()
            theory = output / '박주성_문제1.md'
            theory.write_text('User-authored theory answer; do not modify.', encoding='utf-8')
            report = output / '박주성_문제2.md'
            report.write_text('Separately authored analysis report; do not modify.', encoding='utf-8')
            environment = os.environ.copy()
            environment.pop('PYTHONPATH', None)
            result = subprocess.run(
                [sys.executable, str(ROOT / 'main.py'), '--output-dir', str(output)],
                cwd=temporary, env=environment, capture_output=True, text=True,
                timeout=120, check=True,
            )
            self.assertIn('selected k=3', result.stdout)
            for filename in ['submission_forecast.csv', 'submission_segments.csv']:
                self.assertEqual((output / filename).read_bytes(), (ROOT.parent / filename).read_bytes())
            self.assertEqual(theory.read_text(encoding='utf-8'), 'User-authored theory answer; do not modify.')
            self.assertEqual(report.read_text(encoding='utf-8'), 'Separately authored analysis report; do not modify.')
            forecast = pd.read_csv(output / 'submission_forecast.csv')
            self.assertEqual(list(forecast.columns), ['date', 'predicted_sales'])
            self.assertEqual(len(forecast), 28)
            segments = pd.read_csv(output / 'submission_segments.csv')
            self.assertEqual(list(segments.columns), ['user_id', 'segment'])
            self.assertEqual(segments.segment.value_counts().sort_index().tolist(), [241, 471, 88])
            scores = pd.read_csv(output / 'sourcecode/evaluation/forecast_scores.csv')
            final = scores.loc[scores.evaluation.eq('final')].set_index('model')
            self.assertAlmostEqual(final.loc['ridge_11', 'mape_pct'], 9.9436663828, places=7)
            self.assertAlmostEqual(final.loc['baseline', 'mape_pct'], 10.7282155707, places=7)
            selection = pd.read_csv(output / 'sourcecode/evaluation/cluster_selection.csv')
            self.assertEqual(selection.loc[selection.silhouette.idxmax(), 'k'], 3)
            self.check_outputs(output)

    def check_outputs(self, output):
        self.assertEqual(len(list((output / 'images').glob('*.png'))), 3)
        self.assertFalse(list(output.rglob('*.zip')))
        self.assertFalse((output / '비즈니스_액션플랜.md').exists())

    def test_clean_sources_generate_only_analysis_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / 'sourcecode'
            package.mkdir()
            for name in ['main.py', 'requirements.txt']:
                shutil.copy2(ROOT / name, package / name)
            for name in ['src', 'data', 'config']:
                shutil.copytree(ROOT / name, package / name, ignore=shutil.ignore_patterns('__pycache__'))
            result = subprocess.run([sys.executable, str(package / 'main.py')],
                                    cwd=temporary, capture_output=True, text=True, timeout=120, check=True)
            self.assertIn('Generate analysis figures', result.stdout)
            self.check_outputs(Path(temporary))
            self.assertFalse(list(Path(temporary).rglob('*.md')))
            for filename in ['submission_forecast.csv', 'submission_segments.csv']:
                self.assertEqual((Path(temporary) / filename).read_bytes(), (ROOT.parent / filename).read_bytes())


if __name__ == '__main__':
    unittest.main()
