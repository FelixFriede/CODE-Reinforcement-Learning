# README

Python version

    3.10.13

Runtime dependencies are listed in requirements.txt can be installed via

    source venv/bin/activate
    pip install -r requirements.txt

Always run files from the root folder. For example, run ROOT/playground/hello_world.py via:

    source venv/bin/activate
    python -m playground.hello_world

Each file header contains information about its contents, i.e. if it is CORE (part of the final assigment), LEGACY or TESTING.
Whenever parts of the code (like some LEGACY content and some plots) are purely AI generated and barely proof read, they are marked as such.

### Project structure

    CODE-Reinforcement-Learning/
    │
    ├── src/                  [shared] bandits, algorithms
    ├── script/               [shared] running experiments (Blatt 1,2)
    ├── util/                 [shared] core utility functions
    ├── out/                  [shared] figures, data
    ├── doc/                  [shared] analysis (Blatt 3)
    │
    ├── venv/                 [gitingored] (you create) python virtual environment
    ├── log/                  [gitignored] (some files might create) runtime logs
    │
    ├── requirements.txt      [shared] (you install) dependencies
    ├── prompts.txt           [shared] API and coding style information for AI agents.
    ├── .gitignore            [shared]
    ├── README.md             [shared]

### Important files and their classes

    - [LEGACY] script/Task_1_8_naive
    - [CORE] script/Task_1_8_graphs
    - [CORE] script/Task_1_8_performance
    - [CORE] script/Task_2_5_graphs
    - [TODO] script/Task_2_5_performance
    - [CORE] src/bandits                    [CORE] Gang_of_Bandits, [LEGACY] Bandit
    - [TODO] src/boltzmann
    - [CORE] src/etc                        [CORE] ETCBulkAlgorithm, [LEGACY] ETCAlgorithm
    - [TODO] src/gradient
    - [TODO] src/greedy
    - [TODO] src/ucb

### Features

- Instead of naively pulling one arm at a time, bandits and algorithms support bulk pulling.
  This greatly reduces runtime. However, tracking means and other data becomes somewhat convoluted.
  As this was noticed early, the single pull version of most algorithms where never tested or even human read.
- [UNFINISHED] For fairness, algorithms will be tested on the same seed - For better data, seeds are randomized between runs.
- Parameters are picked out of a coarse grid with 50 candidates via successive 1/3-ing with reduced resource allocation.
  Reducing pull count is questionable, especially for ETC, but seems to be a non-issue.

### Expected runtime

On my laptop:

    Intel(R) Celeron(R) N4500 @ 1.10GHz (2 Cores, 2 Threads)
    RAM: 8 GB
    Linux Mint 22.2

    script/Task_1_8_performance:
    script/Task_2_5_performance:

On some cloud computer:

    Intel(R) Celeron(R) N4500 @ 1.10GHz (2 Cores, 2 Threads)
    RAM: 8 GB
    Linux Mint 22.2

    script/Task_1_8_performance:
    script/Task_2_5_performance:

---

# INTERN

### ToDo

- [TODO] Create flow diagram
- [TODO] Change main such that each algorithm may be commented out individually.
- [TODO] Create documentation of standout choices and features.
- [TODO] Which runs share seeds for fairness? Which benefit from random seeds?
- [DONE] Separate data from plots, i.e. store data separately for easier creating of nicer plots.
- [DONE] Update successive halving: Remove last round, manually update winner return.
- [DONE] Implement smart argument optimizer algorithm.

### Known Problems

- [PROBLEM] Softmax is never used. I believe it should be.
- [PROBLEM/BUG] For some algorithms the parameter edge case is optimal. See below. Change parameter range. May be a bug.
- [BUG] The curated plots do not work.
- [BUG] Tune mean regret and final regret are (for some algorithms) very different. This is likely because the mean is being calculated over an mostly empty array, since only small gangs are part of the trials.
- [BUG] Boltzmann with arbitrary noise does not work. (Or is really bad.)
- [FIXED] plotting in bulk algorithms requires arms to be in order. However, this is obviously a problem, since algorithms except ETC are usually arm order dependent.

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
