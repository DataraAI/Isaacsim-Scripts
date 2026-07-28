# Cable Mount Roll and Forward Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Roll the pregrasped RJ45 connector exactly 90 degrees around its insertion axis and extend the physical RJ45 ToolCenter 30 mm farther from `panda_hand` without corrupting visual-servo standoff or future insertion depth.

**Architecture:** Update the calibrated `panda_hand` to `/World/ToolCenter` transform in `IKConfig`. Leave cable mounting unchanged so the detected RJ45 tip still maps directly onto ToolCenter and the existing fixed joint captures the new real hand-to-plug pose.

**Tech Stack:** Python 3, NumPy, `unittest`, Isaac Sim 6.0.0.

## Global Constraints

- Set ToolCenter local translation to `(0.0, 0.0, 0.1334)` m.
- Set ToolCenter local orientation WXYZ to `(0.7071067811865476, 0.0, 0.0, 0.7071067811865475)`.
- Preserve the RJ45 nose direction along ToolCenter local `+Z`.
- Keep the real RJ45 tip coincident with `/World/ToolCenter`.
- Do not add cable-presentation offsets or change `cable_mount.py`.
- Do not change camera calibration, fixed-joint topology, deformable attachments, visual-servo standoff, or validation limits.

---

### Task 1: Add the failing calibration contract test

**Files:**
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

- [x] Import `numpy` and `CONFIG`.
- [x] Assert `CONFIG.ik.tool_center_local_position_m == (0.0, 0.0, 0.1334)`.
- [x] Assert the scalar-first ToolCenter quaternion equals the positive 90 degree local-Z rotation.
- [x] Assert no `presentation_roll_deg` or `forward_tip_offset_m` settings exist.
- [x] Run the focused test and confirm it fails against the old `0.1034` m identity calibration.

### Task 2: Update the calibrated ToolCenter

**Files:**
- Modify: `single_rack_cv/config.py`

- [x] Change the local Z translation from `0.1034` m to `0.1334` m.
- [x] Change the local orientation from identity to `(0.7071067811865476, 0.0, 0.0, 0.7071067811865475)`.
- [x] Leave `CableMountConfig`, `cable_mount.py`, camera configuration, target orientation, standoff, and tolerances unchanged.
- [x] Run `python -m unittest tests.test_runtime_wiring -v` and confirm all runtime-wiring tests pass.

### Task 3: Repository and workstation verification

**Files:**
- Verify: `single_rack_cv/config.py`
- Verify: `single_rack_cv/tests/test_runtime_wiring.py`

- [x] Run repository-side tests excluding the Isaac-only `pxr` test in the non-Isaac development container.
- [x] Confirm the only production diff is the ToolCenter calibration.
- [ ] On the Isaac workstation, pull `feature/pregrasped-cable-mount` and run:

```bash
cd ~/Isaacsim-Scripts
git switch feature/pregrasped-cable-mount
git pull --ff-only
cd single_rack_cv
~/.local/share/ov/pkg/isaac-sim-6.0.0/python.sh main.py 2>&1 | tee camera_output/cable_mount_roll_offset_console.txt
```

- [ ] Accept only when the viewport shows the 90 degree roll and 30 mm protrusion and the logs still satisfy the strict fixed-joint, attachment, tip-error, and axis-error gates.
