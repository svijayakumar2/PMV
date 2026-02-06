# Prover multi-verifier games

## Installation

Make sure that `uv` is installed via either:

```curl -Ls https://astral.sh/uv/install.sh | sh```

or using Homebrew:

```brew install astral-sh/uv/uv```

Once `uv` is installed, run:

```commandline
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```


## Training procedure

- Initialization:
  - create dataset (`pmv/data/math_dataset.py`)
  - prover (`pmv/models/prover.py`)
  - verifiers (`pmv/models/verifier.py`)
  - aggregator (`pmv/aggregator.py`)
  - TODO: role-policy (`pmv/models/policy.py`) -- may be needed b/c sneaky epidsodes stop working as verifiers improve
- Iterate
  - learn/update aggregator (`train_f` in `pmv/aggregator.py`)
  - train the prover (and role-policy) (`train_prover` in `pmv/train.py`)
  - supervise verifiers (`train_verifiers` in `pmv/train.py`)

## Reference

If you find this code useful in your research, please cite the following:

```commandline
@article{vijayakumar2025prover
}
```
