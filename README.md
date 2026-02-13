# CODE-Reinforcement-Learning

Python version: 3.10.13 (arbitrary for now)

Notice extensive .gitignore - you need to set that up for yourself. I used pyenv, worked great.

Intended foulder structure below. Everything denoted [local] is in the -gitignore

rl-project/
│
├── src/                           [shared]  core project code
│   ├── .../                       [shared]  ...
│
├── docs/                          [shared]  additional analysis & visualization only
│
├── venv/                          [local]   python virtual environment
│
├── submission*/                   [shared]  generated, zip-ready hand-ins      DO NOT TEST HERE
│   └── assignment*/               [shared]  self-contained TA executable code  DO NOT TEST HERE
│
├── logs/                          [local]   runtime logs
├── output/                        [local]   figures, metrics, checkpoints
├── playground/                    [local]   unrelated python testing
│
├── .gitignore                     [shared]  excludes local & generated files
├── requirements.txt               [shared]  pinned runtime dependencies
├── pyproject.toml                 [shared]  tooling & formatting config
├── README.md                      [shared]  dev & workflow documentation

