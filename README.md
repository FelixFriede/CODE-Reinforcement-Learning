# CODE-Reinforcement-Learning

## Setup

_KEEP IN MIND THAT I USE LINUX. THE COMMANDS BELOW MAY OR MAY NOT WORK IN WINDOWS_

---

Python version: 3.10.13 (arbitrary for now)  
Notice extensive .gitignore - you need to set that up for yourself. I used pyenv, that worked fine. See more below.

---

## Rules of Engagement

- Read the README :)
- Respect folder architecture, I do not want to pull random logs
- Please add any python generated files such as caches that I missed to the .gitignore
- Make sure you update requirements.txt

---

## How to run anything.

Create your own venv in the root folder CODE-Reinforcement-Learning.  
I used pyenv, ChatGPT knows how to use that. (The Tutorial on its website is suboptimal)

---

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
    ├── util/                 [shared] core utility functions
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

For specifying files, I recommend using

    from util.io_helpers import ROOT_DIR

then using the path relative to that. Avoid absolute paths for portability. Paths relative to the .py file are fine,
but may break if the file is moved. (for example to the submission folder)

---

Use log() and out() from util. Both can be used with or without specifying the exact file name. Example usage below:

    from util.io_helpers import log, out

    log("This is a test message. Output will be log/log.txt")
    log("This is a test message. Output will be log/my_custom_log.txt", "my_custom_log.txt")

    out("Hello world")
    out("Hello world", "demo.txt")

---

## requirements.txt

Please add any packages you use here, so that others can add them to their venvs:

    numpy==1.24.2          # exactly version 1.24.2
    pandas>=2.0.0          # any version 2.0.0 or newer
    scikit-learn<=1.3.0    # version up to 1.3.0
    matplotlib             # latest version available

To update your venv, then use

    source venv/bin/activate
    pip install -r requirements.txt

To check installed packages, use

    pip list

---
