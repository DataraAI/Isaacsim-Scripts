# Alignment Stress-Test Design

## Purpose

Build a deterministic qualification harness for the existing single-rack RGB stereo visual-servo pipeline. The harness must prove that the controller can recover from a 3x3 grid of world-frame Y/Z starting-position offsets without changing the controller's safety rules or hiding failures behind averages.

This work tests alignment robustness only. It does not add insertion, orientation control, depth-axis start variation, lighting variation, or multi-port selection.

## Existing behavior that must remain unchanged

- Runtime control remains image-only.
- RTX/USD data may be used only for post-run scoring.
- Wrist orientation remains fixed.
- The ToolCenter target step remains limited to 1 mm.
- Failed observations hold position and trigger reacquisition.
- No insertion command is allowed.
- Normal `main.py` behavior remains unchanged when stress-test arguments are absent.

## Stress matrix

Use the current nominal ToolCenter start pose from `CONFIG.ik.initial_position` and apply only world-frame Y/Z offsets.

- X offset: 0 mm for every run.
- Y offsets: -10, 0, +10 mm.
- Z offsets: -10, 0, +10 mm.
- Orientation: unchanged from `CONFIG.ik.initial_orientation_wxyz`.
- Repeats: 3 per pose.

This produces 9 poses and 27 independent runs.

The run order must be deterministic but shuffled using a recorded fixed seed. The seed must be written into the suite summary so the exact order can be reproduced.

## Process isolation

Each run must start a fresh Isaac Sim process. The harness must not reset and reuse one live Isaac process across poses.

Fresh-process isolation is required because the following state can otherwise leak between runs:

- YOLOE detector and prompt state.
- Stereo tracking references.
- Camera and renderer buffers.
- IK and articulation state.
- Visual-servo acquisition and convergence counters.
- Previous failure or reacquisition history.

The suite runner launches runs sequentially. Parallel Isaac processes are out of scope because they would introduce GPU-memory contention and make timing results harder to interpret.

## Components

### `stress_alignment.py`

Pure Python module with no Isaac imports.

Responsibilities:

- Define the canonical 3x3 Y/Z pose matrix.
- Expand the matrix into three repeats per pose.
- Produce deterministic shuffled run order from a fixed seed.
- Define per-run and suite result schemas.
- Evaluate pass/fail gates from one run result.
- Aggregate suite-level metrics and failure groups.

This module must be unit-testable with the normal Python interpreter.

### `main.py`

Add optional command-line stress-run controls while preserving default behavior.

Required arguments:

- `--start-y-offset-mm`
- `--start-z-offset-mm`
- `--stress-run-id`
- `--stress-result-json`
- `--stress-timeout-s`
- `--exit-after-complete`

Behavior when stress arguments are supplied:

1. Create a run-local configuration derived from `CONFIG`.
2. Add the requested world Y/Z offsets to the nominal ToolCenter start position.
3. Keep X and orientation unchanged.
4. Record run metadata and safety counters.
5. Automatically stop after successful visual-servo completion.
6. Stop and report failure when the timeout expires.
7. Always write one result JSON, including on ordinary runtime failure.

Behavior without stress arguments must remain the current interactive runtime.

### `tools/run_alignment_stress.py`

Suite orchestrator.

Responsibilities:

- Build the 27-run matrix from `stress_alignment.py`.
- Launch one fresh `$HOME/isaacsim/python.sh main.py` subprocess per run.
- Pass only explicit stress arguments.
- Capture one complete console log per run.
- Enforce a hard subprocess timeout.
- Continue after ordinary run failures to reveal the full failure pattern.
- Abort only for suite-level infrastructure failures such as a missing Isaac launcher or unwritable output directory.
- Read every run JSON and calculate the final suite result.
- Write `summary.json`, `summary.csv`, and `report.txt`.
- Exit 0 only when all 27 runs pass.
- Exit 2 when the suite completes but one or more runs fail qualification.
- Exit 1 for suite infrastructure failure.

### `tools/run_alignment_stress.sh`

Thin shell entry point that:

- Changes to the project root.
- Sanitizes environment variables that can contaminate Isaac Python.
- Invokes the Python suite runner.
- Preserves the runner's exit status.

## Output layout

Each suite writes into a timestamped directory:

```text
camera_output/alignment_stress/<timestamp>/
  summary.json
  summary.csv
  report.txt
  runs/
    y-10_z-10_repeat-1/
      console.log
      result.json
```

Run directory names must encode Y offset, Z offset, and repeat number. Existing suite directories must never be overwritten.

## Per-run result schema

Each `result.json` must include at least:

- Schema version.
- Run ID.
- Start Y offset in millimeters.
- Start Z offset in millimeters.
- Repeat number.
- Start and end timestamps.
- Duration in seconds.
- Process exit status.
- Completion state.
- Timeout state.
- Track-acquired state.
- Visual-alignment-locked state.
- Final center error in pixels.
- Final range error in millimeters.
- Final physical ToolCenter tracking error in millimeters.
- Final benchmark-ground-truth position error in millimeters.
- Maximum commanded ToolCenter target step in millimeters.
- Maximum orientation deviation in degrees.
- Perception rejection count.
- Track-loss/reacquisition count.
- Fatal error text, if any.
- Confirmation that no insertion command was issued.
- Final list of failed gates.
- Overall qualified boolean.

Missing or non-finite required metrics fail the run.

## Ground-truth scoring

The controller must not read RTX/USD ground truth during motion.

After visual-servo completion, the stress-run path may use the existing benchmark-only ground-truth mechanism to calculate the final estimated opening/target position error. This value is scoring-only and must not affect target commands, reacquisition, or completion logic.

If benchmark ground truth cannot be produced or read, the run fails qualification rather than silently dropping the gate.

## Safety instrumentation

The stress path must record enough evidence to verify existing safety constraints:

- Maximum ToolCenter target step must be measured at every target update.
- Wrist orientation deviation must be measured against the fixed initial orientation.
- A boolean or counter must prove that no insertion command was issued.
- Perception failure handling must continue to hold the current target.

Instrumentation must observe existing commands; it must not introduce a second control path.

## Per-run qualification gates

A run passes only when all gates pass:

- Process exits normally.
- Runtime completes within 240 seconds.
- Track acquisition succeeds.
- Visual alignment locks.
- Final center error is at most 2.0 px.
- Absolute final range error is at most 3.0 mm.
- Final physical ToolCenter tracking error is at most 0.3 mm.
- Final benchmark-ground-truth position error is at most 1.0 mm.
- Maximum commanded ToolCenter target step is at most 1.0 mm, with a small numerical comparison tolerance only.
- Maximum wrist orientation deviation is at most the existing orientation-tracking tolerance and no vision-driven orientation command occurs.
- No insertion command occurs.
- No fatal traceback occurs.
- Required result fields are present and finite.

A wrong-port lock is treated as a ground-truth position failure, not averaged with successful runs.

## Suite qualification

The suite passes only when all 27 runs pass.

```text
required_successes = 27 / 27
```

No mean, median, percentile, or pose-level majority can override a failed run.

The suite report must include:

- Overall qualified status.
- Passed and failed run counts.
- Failures grouped by Y/Z pose.
- Failure reasons grouped by gate.
- Worst center error.
- Worst absolute range error.
- Worst benchmark-ground-truth error.
- Worst physical tracking error.
- Maximum commanded step.
- Maximum orientation deviation.
- Minimum, median, p95, and maximum run duration.
- Perception rejection and reacquisition totals.
- Seed and exact execution order.

## Error handling

- Ordinary run failure: write result JSON, preserve console log, continue to the next run.
- Subprocess timeout: terminate the process, mark the run timed out, preserve partial log, continue.
- Missing result JSON: synthesize a failed run record from subprocess status and log path.
- Malformed result JSON: mark the run failed and preserve the parse error.
- Missing Isaac launcher or unwritable suite directory: abort with infrastructure exit code 1.
- User interrupt: terminate the active child process, write a partial suite report when possible, and exit nonzero.

## Testing

Pure tests must cover:

- Exact 9-pose matrix generation.
- Three repeats per pose.
- Deterministic shuffle and reproducible order.
- Unique run IDs and directory names.
- Every per-run gate boundary.
- Missing and non-finite metric rejection.
- 27/27 suite requirement.
- Failure grouping and worst-case aggregation.
- Exit-code mapping.
- Command construction for Isaac subprocesses.
- Result synthesis for timeout, missing JSON, and malformed JSON.

Structural tests must verify:

- Default `main.py` remains runnable without stress arguments.
- Stress mode does not alter X or orientation.
- No insertion path is added.
- Runtime ground-truth use is restricted to post-completion scoring.
- The existing 1 mm target-step limit remains unchanged.

Workstation validation must include:

1. Existing pure and structural test suite.
2. Existing 60-pair front-plane benchmark with `QUALIFIED=true`.
3. One nominal stress run.
4. Full 27-run stress suite.
5. Inspection of all failed-run logs if qualification is not 27/27.

## Kill switch

Do not proceed to insertion if the stress suite is below 27/27.

If failures cluster at one or more offsets, investigate perception geometry, reachability, or convergence at those poses. Do not weaken gates or remove failing poses to manufacture a pass.

## Explicit non-goals

- No X/depth start offsets.
- No wrist orientation perturbations.
- No lighting or material randomization.
- No different rack or port selection.
- No insertion motion.
- No controller gain tuning as part of the harness implementation.
- No parallel Isaac execution.
