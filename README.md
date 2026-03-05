# CODE-Reinforcement-Learning

    CODE-Reinforcement-Learning/
    │
    ├── src/                  [shared] core project code
    ├── script/               [shared] entry points
    ├── util/                 [shared] core utility functions
    ├── out/                  [shared] figures, metrics, data
    ├── doc/                  [shared] additional analysis & visualization
    │
    ├── venv/                 [local] python virtual environment
    ├── log/                  [local] runtime logs
    ├── playground/           [local] unrelated python testing
    ├── tests/                [local] standard tests.
    │
    ├── .gitignore            [shared] excludes local & generated files
    ├── requirements.txt      [shared] pinned runtime dependencies
    ├── prompts.txt           [shared] API and coding style information for AI agents.
    ├── README.md             [shared]

### ToDo

- [TODO] Change main such that each algorithm may be commented out individually.
- [TODO] Create documentation of standout choices and features.
- [TODO] Which runs share seeds for fairness? Which benefit from random seeds?
- [DONE] Separate data from plots, i.e. store data separately for easier creating of nicer plots.
- [DONE] Update successive halving: Remove last round, manually update winner return.
- [DONE] Implement smart argument optimizer algorithm.

---

### Known Problems

- [PROBLEM] Softmax is never used. I believe it should be.
- [PROBLEM/BUG] For some algorithms the parameter edge case is optimal. See below. Change parameter range. May be a bug.
- [BUG] The curated plots do not work.
- [BUG] Tune mean regret and final regret are (for some algorithms) very different. This is likely because the mean is being calculated over an mostly empty array, since only small gangs are part of the trials.
- [BUG] Boltzmann with arbitrary noise does not work. (Or is really bad.)
- [FIXED] plotting in bulk algorithms requires arms to be in order. However, this is obviously a problem, since algorithms except ETC are usually arm order dependent.

---

### Human read code

- Basically nothing at this point. I'll work on that. Tho most things really seem fine.

---

# Features

- Instead of naively pulling one arm at a time, bandits and algorithms support bulk pulling. This greatly reduces runtime.
- [UNFINISHED] For fairness, algorithms will be tested on the same seed - For better data, seeds are randomized between runs.
- Parameters are picked out of a coarse grid with 50 candidates via successive 1/3-ing with reduced resource allocation.
  Reducing pull count is questionable, especially for ETC, but seems to be a non-issue.

---

# Setup

_Keep in mind that the commands below are for linux mint (thus ubuntu, debian). I dont know how windows works._

---

### How to run anything.

Python version: 3.10.13
Create your own venv in the root folder CODE-Reinforcement-Learning.

---

Once run (obviously with your path)

    cd /home/felix/CODE-Reinforcement-Learning
    source venv/bin/activate

Then always run files from the root folder. For example, run ROOT/playground/hello_world.py via:

    python -m playground.hello_world

---

### Shared utilites

For portability when specifying (output) files, I recommend the path relative to using

    from util.io_helpers import ROOT_DIR

---

Use log() and out() from util. Both can be used with or without specifying the exact file name. Example usage below:

    from util.io_helpers import log, out

    log("This is a test message. Output will be log/log.txt")
    log("This is a test message. Output will be log/my_custom_log.txt", "my_custom_log.txt")

    out("Hello world")
    out("Hello world", "demo.txt")

---

### requirements.txt

Please add any packages you use here, so that others can add them to their venvs. Version may or may not be specified:

    numpy==1.24.2          # exactly version 1.24.2
    matplotlib             # latest version available

To update your venv, then use

    source venv/bin/activate
    pip install -r requirements.txt

---

# Ideal coefficients and their linspace

- ETC 16 (1-50)
- Greedy NONE
- EpsGreedyFixed 0.04 (0.01-0.50)
- EpsGreedyDecreasing 6.21 (0.01-50.00)
- UCB 0.50 (0.01-0.50) [!]
- UCBSubGaussian 0.35 (0.01-0.50)
- BoltzmannSoftmax 0.35 (0.01-0.50)
- BoltzmannGumbel 10.28 (0.01-50.00)
- BoltzmannArbitraryNoise(gumbel) 8.25 (0.01-50.00)
- GumbelScaledBonus 0.01 (0.01-50.00) [!]
- PolicyGradient 0.33 (0.01-0.50)
- PolicyGradientBaseline 0.45 (0.01-0.50)
