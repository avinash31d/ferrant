# Final review fixes report

## Scope completed

- Replaced the 64-bit FNV Git source cache key with SHA-256 over length-delimited, presence-aware repository/ref/native-subdirectory fields.
- Added `.ferrant-source.json`, preserving repository/ref exactly and the native subdirectory bytes as hex, plus the resolved commit. Cache reuse now requires a matching manifest, matching `HEAD`, and no tracked checkout modifications. Invalid entries rebuild through the existing staged rename/backup path.
- Reserved `load_skill` and `read_skill_resource` whenever `.skills()` is enabled. Earlier conflicting tools are removed; later conflicting tools are ignored; generated specs remain unique.
- Documented and enforced the supported resource-read trust model: local skill packages must not be concurrently mutated during a read. Stable traversal/symlink containment checks remain, but the API does not claim adversarial TOCTOU protection.
- Changed the test `TempDir` helper to UUID candidates with exclusive `create_dir` and collision retry.
- Added cache lock owner records (PID, timestamp, nonce), stale-owner recovery, conservative handling of incomplete lock records, and live-owner non-stealing behavior.

## TDD evidence

Regression tests were added before production changes for mismatched manifests, mismatched checkout `HEAD`, exact optional identity representation, stale lock recovery, live lock preservation, and reserved-tool registration in both builder orderings. The initial RED execution was attempted with:

`cargo test --test skills github_rebuilds_cache_when_identity_manifest_is_missing_or_mismatched -- --nocapture`

It reached dependency compilation but the installed GNU Rust toolchain could not find `gcc.exe`/`dlltool.exe`, so the environment prevented an executable RED result. A later full test attempt with the installed MSVC Rust toolchain likewise stopped before compiling Ferrant because `link.exe` is not installed.

## Verification commands and results

- `rustfmt --edition 2021 --check src/agent.rs src/skills.rs tests/skills.rs` — passed.
- `git diff --check` — passed.
- `cargo metadata --no-deps --format-version 1` — passed.
- `cargo +stable-x86_64-pc-windows-msvc test --test skills` — blocked before project compilation: MSVC `link.exe` missing.
- `cargo test --test skills ...` on the default GNU toolchain — blocked before project compilation: `gcc.exe` and `dlltool.exe` missing.

## Concerns

- Runtime tests could not execute in this Windows environment because neither installed Rust target has its required native C/linker toolchain. The regression suite and sources are formatted and the dependency lock was resolved, but compilation/test execution still needs a machine with MinGW GCC/binutils or Visual C++ Build Tools.
- Resource reads intentionally rely on a documented trusted-package/no-concurrent-mutation assumption; they are not presented as secure against an adversary racing filesystem path replacement.
- PID-based stale detection is necessarily subject to OS PID reuse; the lock record also carries timestamp and nonce for ownership/auditability, and incomplete records are never stolen.
