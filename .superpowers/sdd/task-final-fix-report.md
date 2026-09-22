# UR5e 6x Cable Insertions — Final Review Fix Report

Date: 2026-09-13
Branch: `feature/ur5e-6x-cable-insertions`

## Fix notes

- C1: Lowered `MATING_SIDE_MARGIN_M` from 0.0005 m to 0.00015 m. Added a
  host regression test that loads `E_crystal_head1_45` and `jack_upper_c2`
  through `insertion_features.cache`, poses the head at a nominal mate, and
  verifies `evaluate_alignment(...).passed` using shipped constants.
- I1: Latch Z contributes to `pos_error_m` only while `latch_z_ok` is false,
  preventing continued Z nudges after clearance has passed.
- I2: Port standoff now uses
  `mating_center - standoff * unit(insertion_axis)` in both queueing and
  fallback validation. Replaced the fixed-world-X source assertion with
  behavior coverage for a rotated cached port.
- I3: `_live_features` validates that both live feature transforms have
  unit-length linear columns before applying cached metre-scale geometry.
  Scaled transforms now raise a descriptive `ValueError`.
- I5: End-effector frame selection recursively searches the robot hierarchy
  for `tool0`, falls back to `wrist_3_link`, and logs the selected frame.
- Defined `GRASP_RELEASE_WAIT_FRAMES = 90` in config and removed the runtime
  default.
- README now states that debug marker prim creation is not implemented and
  documents the axis-relative standoff.
- Removed the duplicate align wrapper hold monitor; `tick_align_and_insert`
  is now the single align-phase monitor call site.

## Verification

Command:

```text
cd aayush && python3 -m unittest discover -s ur5e_6x_cable_insertions/tests -v
```

Full output:

```text
test_aligned_features_pass (test_alignment.AlignmentTests.test_aligned_features_pass) ... ok
test_clamp_nudge_limits_step (test_alignment.AlignmentTests.test_clamp_nudge_limits_step) ... ok
test_insert_step_moves_along_port_axis (test_alignment.AlignmentTests.test_insert_step_moves_along_port_axis) ... ok
test_latch_z_does_not_nudge_after_clearance_passes (test_alignment.AlignmentTests.test_latch_z_does_not_nudge_after_clearance_passes) ... ok
test_mating_gap_along_axis (test_alignment.AlignmentTests.test_mating_gap_along_axis) ... ok
test_port_standoff_follows_negative_insertion_axis (test_alignment.AlignmentTests.test_port_standoff_follows_negative_insertion_axis) ... ok
test_real_caches_pass_at_nominal_mate_with_shipped_constants (test_alignment.AlignmentTests.test_real_caches_pass_at_nominal_mate_with_shipped_constants) ... ok
test_scaled_linear_transform_fails_loudly (test_alignment.AlignmentTests.test_scaled_linear_transform_fails_loudly) ... ok
test_controller_supports_direct_joint_waypoints (test_runtime.ControllerWiringTests.test_controller_supports_direct_joint_waypoints) ... ok
test_controller_uses_ur5e_arm_joint_names (test_runtime.ControllerWiringTests.test_controller_uses_ur5e_arm_joint_names) ... ok
test_align_monitor_has_one_call_site (test_runtime.MainSourceTests.test_align_monitor_has_one_call_site) ... ok
test_align_insert_interleaves_nudges_and_micro_steps (test_runtime.PrimitiveSourceTests.test_align_insert_interleaves_nudges_and_micro_steps) ... ok
test_align_insert_uses_feature_cache_helpers (test_runtime.PrimitiveSourceTests.test_align_insert_uses_feature_cache_helpers) ... ok
test_exports_all_task_6_primitives (test_runtime.PrimitiveSourceTests.test_exports_all_task_6_primitives) ... ok
test_live_features_rejects_scaled_transforms (test_runtime.PrimitiveSourceTests.test_live_features_rejects_scaled_transforms) ... ok
test_no_fixed_joint_weld (test_runtime.PrimitiveSourceTests.test_no_fixed_joint_weld) ... ok
test_port_offset_uses_feature_mating_center_and_insertion_axis (test_runtime.PrimitiveSourceTests.test_port_offset_uses_feature_mating_center_and_insertion_axis) ... ok
test_release_opens_and_home_joint_interpolates (test_runtime.PrimitiveSourceTests.test_release_opens_and_home_joint_interpolates) ... ok
test_observe_hand_computed_from_path39 (test_runtime.SceneWiringTests.test_observe_hand_computed_from_path39) ... ok
test_scene_does_not_fake_grasp_with_fixed_joint (test_runtime.SceneWiringTests.test_scene_does_not_fake_grasp_with_fixed_joint) ... ok
test_scene_wires_ur5e_cable_physics_and_friction (test_runtime.SceneWiringTests.test_scene_wires_ur5e_cable_physics_and_friction) ... ok
test_grasp_tilt_and_friction_match_1x (test_runtime.StationTableTests.test_grasp_tilt_and_friction_match_1x) ... ok
test_home_arm_is_length_6 (test_runtime.StationTableTests.test_home_arm_is_length_6) ... ok
test_per_station_jack_override (test_runtime.StationTableTests.test_per_station_jack_override) ... ok
test_six_stations_and_default_jack (test_runtime.StationTableTests.test_six_stations_and_default_jack) ... ok
test_tree_covers_align_insert_release_home (test_runtime.TaskIntelligenceTests.test_tree_covers_align_insert_release_home) ... ok

----------------------------------------------------------------------
Ran 26 tests in 0.013s

OK
```

## Remaining known issues

- Host tests do not launch Isaac Sim; live six-station physics/IK execution
  remains a runtime validation step.
- Debug marker configuration remains reserved only; marker prim spawning is
  intentionally documented as not implemented.
