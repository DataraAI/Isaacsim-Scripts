# Alignment Stress-Test Design

## Purpose

Build a deterministic qualification harness for the existing single-rack RGB stereo visual-servo pipeline. The harness must prove that the controller can recover from a 3x3 grid of world-frame Y/Z starting-position offsets without changing the controller's safety rules or hiding failures behind averages.

This work tests alignment robustness only. It does not add insertion, orientation control, depth-axis start variation, lighting variation, or multi-port selection.

## Existing behavior that must remain unchanged

- Runtime control remains image-only.
- RTX/USD data may be used only outside the live control loop for scoring.
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

The run order must be deterministic but shuffled with seed `20260722`. The seed and exact execution order must be written into the suite summary so the order can be reproduced exactly.

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
- Produce deterministic shuffled run order from seed `20260722`.
- Define child-runtime, finalized-run, and suite result schemas.
- Evaluate pass/fail gates from one finalized run result.
- Aggregate suite-level metrics and failure groups.
- Compute the expected pre-insert ToolCenter target from canonical benchmark ground truth.

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
4. Record runtime metadata, final poses, and safety counters.
5. Automatically stop after successful visual-servo completion.
6. Stop and report failure when the internal runtime timeout expires.
7. Always write one child-runtime result JSON, including on ordinary runtime failure.

The child-runtime result does not claim its own final process exit status because that is only known reliably by the parent after the child terminates. The suite runner adds the observed subprocess exit status and performs final qualification.

Behavior without stress arguments must remain the current interactive runtime.

### `tools/run_alignment_stress.py`

Suite orchestrator.

Responsibilities:

- Build the 27-run matrix from `stress_alignment.py`.
- Validate the canonical benchmark ground-truth file before launching runs.
- Launch one fresh `$HOME/isaacsim/python.sh main.py` subprocess per run.
- Pass only explicit stress arguments.
- Capture one complete console log per run.
- Set the child's internal runtime timeout to 240 seconds.
- Enforce a 270-second hard parent timeout, allowing 30 seconds for child shutdown and result flushing.
- Continue after ordinary run failures to reveal the full failure pattern.
- Abort only for suite-level infrastructure failures such as a missing Isaac launcher, invalid ground truth, or unwritable output directory.
- Read each child-runtime result, add subprocess metadata, calculate ground-truth error, and evaluate final gates.
- Write the finalized `result.json` for each run.
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
      child_result.json
      result.json
```

Run directory names must encode Y offset, Z offset, and repeat number. Existing suite directories must never be overwritten.

## Result schemas

### Child-runtime result

Each `child_result.json` must include at least:

- Schema version.
- Run ID.
- Start Y offset in millimeters.
- Start Z offset in millimeters.
- Repeat number.
- Start and end timestamps.
- Runtime duration in seconds.
- Completion state.
- Internal-timeout state.
- Track-acquired state.
- Visual-alignment-locked state.
- Final center error in pixels.
- Final range error in millimeters.
- Final ToolCenter target world position.
- Final actual ToolCenter world position.
- Final physical ToolCenter tracking error in millimeters.
- Maximum commanded ToolCenter target step in millimeters.
- Maximum wrist-orientation deviation in degrees.
- Perception rejection count.
- Track-loss/reacquisition count.
- Fatal error text, if any.
- Confirmation that no insertion command was issued.

### Finalized run result

The parent writes `result.json` by preserving child fields and adding:

- Observed subprocess exit status.
- Parent hard-timeout state.
- Console-log path.
- Child-result parse status.
- Expected pre-insert ToolCenter target world position.
- Ground-truth target error in millimeters.
- Final list of failed gates.
- Overall qualified boolean.

Missing or non-finite required metrics fail the run.

## Ground-truth scoring

The live child process must not read RTX/USD ground truth at any time.

Before launching the suite, the parent validates `benchmarks/front_plane_ground_truth.json`. The file must contain the canonical selected port's physical opening center and front-plane normal for the unchanged rack scene.

The parent derives the expected ToolCenter pre-insert target as follows:

1. Read the ground-truth physical opening center.
2. Read and normalize the ground-truth outward front-plane normal.
3. Apply the existing 50 mm pre-insert standoff along that normal using the same sign convention as the current controller.
4. Compare the child's final ToolCenter target world position with that expected pre-insert target.

This comparison happens only after the child exits. It cannot affect motion, reacquisition, completion, or target commands.

If canonical ground truth is missing, malformed, stale for the current scene, or lacks the required center/normal fields, the suite aborts with infrastructure exit code 1 rather than silently dropping the gate.

## Safety instrumentation

The stress path must record enough evidence to verify existing safety constraints:

- Maximum ToolCenter target step must be measured at every target update.
- Wrist orientation deviation must be measured as quaternion angular distance from the fixed initial orientation.
- A boolean or counter must prove that no insertion command was issued.
- Perception failure handling must continue to hold the current target.

Instrumentation must observe existing commands; it must not introduce a second control path.

## Per-run qualification gates

A run passes only when all gates pass:

- Subprocess exits with status 0.
- Internal runtime timeout does not occur.
- Parent hard timeout does not occur.
- Runtime completes within 240 seconds.
- Track acquisition succeeds.
- Visual alignment locks.
- Final center error is at most 2.0 px.
- Absolute final range error is at most 3.0 mm.
- Final physical ToolCenter tracking error is at most 0.3 mm.
- Final ground-truth pre-insert target error is at most 1.0 mm.
- Maximum commanded ToolCenter target step is at most 1.000001 mm, which is the existing 1 mm limit plus a 1e-9 m numerical comparison epsilon.
- Maximum wrist-orientation deviation is at most 0.572958 degrees, equivalent to the existing 0.01 rad orientation tolerance.
- No vision-driven orientation command occurs.
- No insertion command occurs.
- No fatal traceback occurs.
- Required result fields are present and finite.

A wrong-port lock is treated as a ground-truth target-position failure, not averaged with successful runs.

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
- Worst ground-truth target error.
- Worst physical tracking error.
- Maximum commanded step.
- Maximum orientation deviation.
- Minimum, median, p95, and maximum run duration.
- Perception rejection and reacquisition totals.
- Seed and exact execution order.

## Error handling

- Ordinary run failure: write available child data, preserve console log, finalize a failed result, and continue.
- Internal runtime timeout: child writes failure data and exits; parent finalizes the failure and continues.
- Parent hard timeout: terminate the process group, mark the run timed out, preserve partial log, synthesize/finalize failure data, and continue.
- Missing child result: synthesize a failed run record from subprocess status and log path.
- Malformed child result: mark the run failed and preserve the parse error.
- Missing Isaac launcher, invalid ground truth, or unwritable suite directory: abort with infrastructure exit code 1.
- User interrupt: terminate the active child process group, write a partial suite report when possible, and exit nonzero.

## Testing

Pure tests must cover:

- Exact 9-pose matrix generation.
- Three repeats per pose.
- Deterministic shuffle with seed `20260722` and reproducible order.
- Unique run IDs and directory names.
- Every per-run gate boundary.
- Missing and non-finite metric rejection.
- 27/27 suite requirement.
- Failure grouping and worst-case aggregation.
- Exit-code mapping.
- Command construction for Isaac subprocesses.
- Ground-truth expected-target calculation and sign convention.
- Result finalization for timeout, missing JSON, and malformed JSON.

Structural tests must verify:

- Default `main.py` remains runnable without stress arguments.
- Stress mode does not alter X or orientation.
- No insertion path is added.
- Live child runtime never imports or reads benchmark ground truth.
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
