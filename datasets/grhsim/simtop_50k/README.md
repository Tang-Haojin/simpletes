# GrhSIM SimTop 50k auto-research bench

This bench evolves a structured JSON candidate whose payload is a safe unified
diff against the pinned `wolvrix` revision. The evaluator never edits the user
checkout: it creates a locked local clone, verifies parent `b90d204` and
`wolvrix` `f17e90e`, applies the patch only inside that slot, checks default-off
generated-source identity, builds a real local ELF, runs fixed-ASLR function
gates, and hands resolved artifacts to the trusted runtime.

The score is `control_mean_walltime_ms / candidate_mean_walltime_ms`. Absolute
control and candidate `Host time spent` values and their millisecond/percentage
delta are always returned. CPU, CCD, NUMA, PMU, generated C++, ELF, and profile
data are diagnostics, not substitute objectives. Retryable host-load or audit
failures are retried as the same candidate and do not count toward the valid
candidate budget.

## Start or resume

The launcher fixes the requested model to `gpt-5.6-sol` with reasoning effort
`ultra`, uses four RPUCG chains with `k=1`, serial generation/evaluation, stops
after eight valid generated candidates or sixteen proposals, and disables
reflection. It sources the target checkout's `env.sh` before executing
SimpleTES and forces default-off verification, focused tests, and function
gates on, regardless of inherited shell settings. It passes only the paths of
the MJY configuration files; their
contents are copied to the backend's private temporary `CODEX_HOME` and are not
placed in argv, environment variables, logs, metrics, or checkpoints.

```bash
cd SimpleTES
python datasets/grhsim/simtop_50k/launcher.py --dry-run
python datasets/grhsim/simtop_50k/launcher.py
```

Defaults:

- target checkout: sibling `wolvrix-playground-gsim-calibrate-5`
- config: `~/.codex/config.mjy.toml`
- auth: `~/.codex/auth.mjy.json`
- evaluator slots: `/tmp/simpletes-grhsim-simtop-50k`
- checkpoints: `SimpleTES/checkpoints/grhsim_simtop_50k/<timestamp>`

Use `--resume CHECKPOINT`, and keep the same target/slot roots, to continue a
gracefully stopped run. Build/perf/staging/checkpoint artifacts are deliberately
untracked.

## Candidate and promotion rules

`candidate.schema.json` is the model output contract. Patch paths are limited
to existing tracked C/C++ source/header files under `wolvrix/lib`,
`wolvrix/include`, and `wolvrix/app/pybind`; additions, deletions, binary,
symlink, executable, rename, traversal, generated-output, submodule, harness,
build-description, test, and secret-bearing patches are rejected before
`git apply`. Model output represents `enable_options` as name/value entries;
the evaluator normalizes them into an explicit pinned optimization allowlist.
resume, stats, profile, stop, export, instrumentation, measurement, and
diagnostic policy values are unavailable.

Every generated candidate must be explicitly enabled and byte-identical to the
current-default generated C++/headers while disabled. An additional unpatched
build with the same options must differ from the enabled patched output, so an
old knob alone cannot be credited to a no-op patch. The runtime performs a
quiet, dynamically selected CCD-local ABBA 50k screen with ASLR disabled. A
positive screen is repeated in reversed BAAB order before its pooled score is
used. Only candidates with consistent positive walltime are eligible for later
code retention/default promotion. Promotion documentation and commits belong
in `pdocs/grhsim_opt_thj` under its `RULES.md`; evaluator artifacts themselves
must not be committed.

The cached control is reused only when its pinned revisions, copied `env.sh`,
fixed build configuration, resolved toolchain fingerprint, generated-source
fingerprint, and SHA-256 identities of the ELF, image, and NEMU all match.
