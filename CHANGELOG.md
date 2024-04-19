# Changelog

All changes we make to the assignment code or PDF will be documented in this file.



## [unreleased] - yyyy-mm-dd

### Added

### Changed

- code: remove `humanfriendly` from requirements.txt, add `matplotlib`
- handout: modify problem `distributed_communication_multi_node` to specify that
  multinode measurements should be 2x1, 2x2, and 2x3.

### Fixed

- handout: fixed cut off text in problem memory_profiling (a)
- handout: fixed mismatch between slurm config and description text in section 3.2

## [0.0.1] - 2024-04-17

### Added

- handout: added a short question about variability on problem benchmarking_script

### Changed

### Fixed

- handout: fixed typo in problem `triton_rmsnorm_forward`. The adapters should
  return the classes, not the `.apply` attribute.
- code: added `-e` flag to `./cs336-systems/'[test]'`
- handout: clarified recommendation about the timeit module
- handout: clarified question about kernel with highest CUDA total

## [0.0.0] - 2024-04-16

Initial release.
