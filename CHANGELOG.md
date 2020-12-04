# Changelog
**Main changes** to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.0] - 2020-12-04
### Changed 
- How non-default levels file is read, see #23. This will break compatibility with earlier versions.

### Fixed
- Fix of error message for bExSingle, with sideeffect of slight changes in the random numbers drawn for that population mode, see #22.
- updated briccs to 64bit

## [1.5.0] - 2020-02-27
### Fixed
- Bugfix for Wigner distribution (Wigner distribution was set up incorrect.) see 46c7841

## [1.4.1] - 2019-01-15
### Added
- Include DOI for the repository

## [1.4.0] - 2019-01-15
### Added
- option to use tabulated GSF
- option to use (spin)-parity dependent population cross-section
- checkpoints if program is killed early and possibility to save to Root Tree
- license file

## [1.3.0] - 2018-07-02
### Changed
-  Changed ordering and created seperate input parameter file: `settings.h`

### Added
- Implemented the functions GetGg and GetD0 to retreive the average total radiative width and the D0 level spacing of a realization
- Add `g_h2JIntrins`, 2d hist of underlying J dis, for `FullExn` population mode

## [1.2.0] - 2018-05-07
As extracted from the zip files provided by Leo Kirsch

## [1.1.4] - 2018-02-22
As extracted from the zip files provided by Leo Kirsch

[Unreleased]: https://github.com/fzeiser/RAINIER/compare/v1.1.4...HEAD
[1.6.0]: https://github.com/fzeiser/RAINIER/compare/v1.6.0...v1.6.0
[1.5.0]: https://github.com/fzeiser/RAINIER/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/fzeiser/RAINIER/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/fzeiser/RAINIER/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/fzeiser/RAINIER/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/fzeiser/RAINIER/compare/v1.1.4...v1.2.0
[1.1.4]: https://github.com/fzeiser/RAINIER/releases/tag/v1.1.4
