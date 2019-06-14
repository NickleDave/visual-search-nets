# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]
This is the version used for paper submitted to ccneuro 2019
### Added
- DOI badge to README.md
- `tf.data.dataset` pipeline to be able to use larger training data
- ability to 'shard' datasets to use even larger training data
- paper in ``./docs/ccneuro2019`
- scripts for making visual search stimuli in ``./src/bin`
- configs for new training (not fine-tuning) and training on multiple
  stimuli
- ability to specify types of visual search stimuli to use when a single
  run of `searchstims` places paths to all types in a single `json` file
  + using `stim_types` option in `config.ini` file
- ability to specify number of samples per (viusal search stimulus)
  "set size" in training, validation, and test sets
  + enables "balancing" data set
- sub-module for running learning curves (needs to be updated to use
  additions to `searchstims.train`)

### Fixed
- `searchnets.train` uses `MomentumOptimizer` like original AlexNet and
  VGG16 papers

### Changed
- how `searchnets.test` saves results file; includes name of 'config.ini`
in saved filename

## [0.1.0]
### Added
- made into separate project
- set up repository with structure from "Good Enough Practices for Scientific Computing"
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510#sec009>
- add ability to run learning curve to test whether train / test accuracy are converging
  + i.e. would we benefit from more training data
- can compute accuracy per epoch per set size on training set, saves to .txt file

### Fixed
- `__main__` checks whether config file exists and throws a human-readable error if it doesn't,
as opposed to failing weirdly because `ConfigParser` doesn't tell you when the `.ini` file does not exist
