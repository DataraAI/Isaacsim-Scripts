# single_rack_cv Layout Scope Verification

Date: 2026-08-07

Compared `refactor/single-rack-layout` against qualified baseline `636d4f8a79f021b8e3c73f4dfc726c9148654534`.

- Every changed path is below `single_rack_cv/`.
- `git diff --check` passes.
- The top level contains exactly `main.py`, `sim.py`, `config.py`, and `debug.py` as Python files.
- The old `cable_runtime.py` / `cable_runtime/` collision is absent.
- `runtime/cable_runtime_base.py`, `runtime/cable_runtime.py`, and `runtime/full_insertion_runtime.py` are present.

Merge remains blocked on the real Isaac Sim workstation qualification of the exact PR head.
