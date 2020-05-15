# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `transforms` sub-package [#58](https://github.com/NickleDave/visual-search-nets/pull/58)
  + decouples transforms from datasets
  + documents transforms used with each combination of datasets and loss function, in `util.get_transforms` function  
- transforms for VOC target that pick either largest class (based on bounding box)
  or a random bounding box [#59](https://github.com/NickleDave/visual-search-nets/pull/59)
- other `CORnet` models [#66](https://github.com/NickleDave/visual-search-nets/pull/66)
- `tensorboard` module with functions for converting tensorboard events files to 
  Pandas `DataFrame`s and `.csv` files [#69](https://github.com/NickleDave/visual-search-nets/pull/69)
- `analysis` sub-package with functions used to analyze results from 
  `searchstims` and `VSD` experiments
  [#71](https://github.com/NickleDave/visual-search-nets/pull/71)

### Changed
- change `dataset`s so they return dictionaries, enabling more 
  flexibility in what items are present / used for training and testing
  [#67](https://github.com/NickleDave/visual-search-nets/pull/67)
- change how `AbstractTrainer` and `Tester` compute metrics 
  for the `VOCDetection` dataset;  
  now compute *all* metrics during validation and testing
  [#66](https://github.com/NickleDave/visual-search-nets/pull/66)
- remove `utils.munge` and `utils.metrics` modules, refactor 
  functions from them into `analysis` sub-package
  [#71](https://github.com/NickleDave/visual-search-nets/pull/71)

### Fixed
- now actually using the correct target corresponding to different loss 
  functions when training on `VOCDetection` dataset, e.g. `CE-largest` 
  uses `largest` from the batch dictionary
  [#66](https://github.com/NickleDave/visual-search-nets/pull/66)

### Removed
- automatic setting of defaults for plotting, that were 
  in `plot.__init__` which made it hard to override them
  [#72](https://github.com/NickleDave/visual-search-nets/pull/72)

## [1.0.0]
This is the version used for SfN 2019 poster, and for the paper
### Added
- logging to a `tf.events` file [#49](https://github.com/NickleDave/visual-search-nets/pull/49)

### Fixed
- fix checkpoint saving [#46](https://github.com/NickleDave/visual-search-nets/pull/46)
  + save "best" checkpoint as well as intermittent / last epoch checkpoint
  + don't save a separate model file

### Changed
- switched to `torch` [#33](https://github.com/NickleDave/visual-search-nets/pull/33)
- use `pyprojroot` [#45](https://github.com/NickleDave/visual-search-nets/pull/33)
- clean up codebase [#44](https://github.com/NickleDave/visual-search-nets/pull/33)
- rename `classes` subpackage to `engine` [#48](https://github.com/NickleDave/visual-search-nets/pull/48)

## [0.3.0]
This is the version used for presentation at SciPy 2019
### Added
- functionality in `utils.results_csv` that computes d prime and adds it
  to the .csv, as well as accuracy across both target present and target
  absent conditions (i.e. what most people would call just "accuracy")
- single-source version
- summary results files and files with paths to training/validation/test
  data are part of repository

### Changed
- `figures.acc_v_set_size` re-written as more general `metric_v_set_size`,
  works with d-prime metric and can plot accuracy, means, etc., for both
  conditions (instead of always separating out target present and target
  absent conditions into two lines)

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
- ability to specify number of samples per (visual search stimulus)
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
