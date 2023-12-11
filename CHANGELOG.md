# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.8] - 2023-12-11

### Added

- `Nx.argsort` supports tensors rank = 2 (#66)
- `Nx.dot` supports batched axes (#68)
- `Nx.window_sum` support (#71)
- `Nx.dot` more complete support for M x N tensors operation (#73)

## [0.1.7] - 2023-12-01

### Added

- `Nx.put_slice`, more complete support (#46)
- `Nx.dot`, support n x m (#51)
- `Nx.take` supports indices rank > 1 (#57)
- `Nx.argsort` basic support, only CPU (#59)
- `Nx.pad`, support tensors of rank > 1 (#61)
- `Nx.pad` supports negative padding (#62)
- `Nx.window_max` supports `:same` padding (#64)

## [0.1.6] - 2023-11-24

### Added

- Nx.conv: partially support padding and strides options
- Nx.window_max: initial partial support

## [0.1.5] - 2023-11-20

### Added

- Precompiled binary for ARMv6 gnueabihf
- Support Nx.reverse

### Fixed

- Fixes wrong behavior in Nx.slice when passing Nx.Tensor as start_indices

## [0.1.4] - 2023-11-06

### Fixed

- Precompiled binaries name fix

## [0.1.3] - 2023-11-06

### Added

- Precompiled binary for CUDA 12.x target.

## [0.1.2] - 2023-10-30

### Added

- Precompiled binaries for a few CPU targets.
