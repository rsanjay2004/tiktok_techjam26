# KuaiRand Starter Kit Test Report

Tested with:

- Source zip: `/Users/livelysan/Downloads/kuairand-starter-kit.zip`
- Extracted test copy: `/tmp/kuairand-test/kuairand-starter-kit`
- Python: 3.13.3
- Temporary dependency: `numpy==2.5.2`

## What the zip contains

The zip contains the benchmark harness, not the real KuaiRand-Pure dataset:

- `README.md`
- `data.py`
- `evaluate.py`
- `baseline.py`
- `submit.py`
- `ablation_features.py`
- `baseline_scores.json`

The real benchmark still requires the downloaded `KuaiRand-Pure/data` CSV files.

## Smoke fixture

Because the real dataset was not included in the zip, a tiny KuaiRand-shaped fixture was created under:

`/tmp/kuairand-test/kuairand-starter-kit/KuaiRand-Pure/data`

The fixture includes the expected file names and columns:

- `video_features_basic_pure.csv`
- `user_features_pure.csv`
- `log_standard_4_08_to_4_21_pure.csv`
- `log_standard_4_22_to_5_08_pure.csv`

It covers the official train/valid/test date ranges.

## Commands tested

```bash
/tmp/kuairand-venv/bin/python -m py_compile baseline.py data.py evaluate.py submit.py ablation_features.py
/tmp/kuairand-venv/bin/python baseline.py --model random
/tmp/kuairand-venv/bin/python baseline.py --model pop
/tmp/kuairand-venv/bin/python baseline.py --model fm --epochs 2 --k 4
/tmp/kuairand-venv/bin/python submit.py --make --split valid /tmp/kuairand-test/submission_valid.csv
/tmp/kuairand-venv/bin/python submit.py --make --split test /tmp/kuairand-test/submission_test.csv
/tmp/kuairand-venv/bin/python submit.py --check --split test /tmp/kuairand-test/submission_test.csv
/tmp/kuairand-venv/bin/python submit.py --score --split valid /tmp/kuairand-test/submission_valid.csv
```

## Results

All smoke tests passed.

The synthetic fixture produced these example results:

| Model | Split | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|---:|
| random | valid | 0.6667 | 0.8770 | 0.7718 |
| random | test | 0.6667 | 0.8770 | 0.7718 |
| pop | valid | 0.5000 | 0.8770 | 0.6885 |
| pop | test | 1.0000 | 1.0000 | 1.0000 |
| fm, 2 epochs | valid | 0.6667 | 0.8770 | 0.7718 |
| fm, 2 epochs | test | 1.0000 | 1.0000 | 1.0000 |

These numbers are only smoke-test numbers from six validation rows and six test rows. They are not meaningful benchmark scores.

## Submission validation

The starter kit generated and validated CSV submissions with the required schema:

```csv
row_id,user_id,video_id,score
```

The test submission check passed:

```text
format and alignment check passed: 6 rows, split=test
```

## Remaining requirement for real benchmarking

To run the real challenge benchmark, download and extract KuaiRand-Pure so this path exists:

```text
KuaiRand-Pure/data/
```

Then run:

```bash
python3 baseline.py --model random
python3 baseline.py --model fm
python3 submit.py --make --split test submission.csv
python3 submit.py --check --split test submission.csv
python3 submit.py --score --split valid submission.csv
```

Expected self-check from the official README:

- random primary should be about `0.475 +/- 0.001`
- FM official baseline test primary is `0.5946`

The project should only claim benchmark improvement after running on the real KuaiRand-Pure dataset.
