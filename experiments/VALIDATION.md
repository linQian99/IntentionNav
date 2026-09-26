# Source Synchronization Validation

Checked on 2026-09-26.

- 180 executed source files match the recorded original source hashes and their
  anonymized public archive copies; two portable recomputation scripts also
  match the public releases.
- `python3 scripts/verify_experiment_sources.py` verifies all 182 checked-in
  source/scorer files against `SOURCE_MANIFEST.json`.
- Python syntax checks pass for 169 files, including the source verifier;
  `bash -n` passes for all three archived shell launchers.
- The shared-category scorer recomputes all 2,320 full/development records with
  matching expected labels, including the 2,000 full-set GSR labels.
- The hosted scorer verifies all 480 records and fresh-plan receipts, and
  reproduces all saved statistics. Within/across-expression disagreement is
  9.1667% / 22.7778%.
- All four existing fresh-planner regression tests pass after binding the
  archived test runner's `SOURCE` to the checked-in `hosted_repeats/code/eval/`
  location. They cover cache clearing, missing calls, cached calls, and receipt
  consistency/tampering.
- Source and configuration files were checked for private workstation/user
  identifiers and credential-like values.

Validation used the previously downloaded public archives and CPU-only checks.
No navigation episodes or hosted API calls were rerun. The archived launchers
retain the original execution structure; new navigation on another machine
requires configuring the paths and runtime dependencies described in README.md.
