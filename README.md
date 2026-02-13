# CODE-Reinforcement-Learning

## Setup

Python version: 3.10.13 (arbitrary for now)
Notice extensive .gitignore - you need to set that up for yourself. I used pyenv, that worked fine.

---

## Rules of Engagement

- Read the README :)
- Respect folder architecture, I do not want to pull random logs

---

## How to run anything.

Once run (obviously with your path)

    cd /home/felix/CODE-Reinforcement-Learning
    source venv/bin/activate

Then, if the file is ./playground/hello_world.py, run:

    python -m playground.hello_world

This is needed for log()/out() to work. As I have absolutely no idea what I am doing, this surely can be improved.
If you do not want to use my log()/out(), you should be able to simply run

    python hello_world

from playground/ or whereever your file is.

---

## Intended foulder structure

See below. Everything denoted [local] is in the .gitignore.

    CODE-Reinforcement-Learning/
    │
    ├── src/                  [shared] core project code
    │ ├── .../                [shared] ...
    │
    ├── doc/                  [shared] additional analysis & visualization only
    │
    ├── venv/                 [local] python virtual environment
    │
    ├── submissions/          [shared] zip-ready hand-ins  DO NOT TEST HERE
    │ └── assignment1/        [shared] ...                 DO NOT TEST HERE
    │
    ├── log/                  [local] runtime logs
    ├── out/                  [local] figures, metrics, checkpoints
    ├── playground/           [local] unrelated python testing
    │
    ├── .gitignore            [shared] excludes local & generated files
    ├── requirements.txt      [shared] pinned runtime dependencies
    ├── pyproject.toml        [shared] tooling & formatting config
    ├── README.md             [shared]

---

## Shared utilites

Use log() and out() from util. Both can be used with or without specifying the exact file name. Example usage below:

    from util.io_helpers import log, out

    log("This is a test message. Output will be log/log.txt")
    log("This is a test message. Output will be log/my_custom_log.txt", "my_custom_log.txt")

    out("Hello world")
    out("Hello world", "demo.txt")
