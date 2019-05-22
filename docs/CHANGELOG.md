# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
