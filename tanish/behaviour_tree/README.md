# Generated behaviour trees in Isaac Sim

## Run the Isaac Sim demo

From the `Isaacsim-Scripts` directory, use Isaac Sim's Python launcher:

```bash
/home/advaith/isaacsim/python.sh tanish/behaviour_tree/isaac_sim_demo.py
```

The demo opens Isaac Sim, creates a Franka, a physics-enabled blue block, and a
green placement goal. It automatically runs the timeline. The Franka physically
closes its fingers on the block, lifts it, carries it, releases it on the green
goal, and verifies the final block pose. The terminal must finish with:

```text
[BT DEMO PASS] Completed 6 generated steps ...
```

This is the tree executed by the bundled JSON:

```text
Sequence: Physical block pick-and-place
├── Action: Move to observation pose
├── Selector: Find the block
│   ├── Condition: Use cached detection
│   └── Retry(2): Detect physics block
├── Selector: Acquire the block
│   ├── Condition: Block is already held
│   └── Retry(2): Physically grasp and lift block
├── Condition: Confirm grasp before transport
├── Action: Carry and release block in green goal
└── Parallel(all): final block-pose and robot-state checks
```

The demo deliberately makes the first perception call fail. This causes the
retry node to execute a second attempt. The two selector nodes also visibly
fall back when no cached detection or existing grasp is present. Look for
`[BT CONDITION]`, `[BT SELECTOR]`, `[BT RETRY]`, and `[BT ACTION]` in the
terminal to follow tree decisions.

For a non-graphical test:

```bash
/home/advaith/isaacsim/python.sh tanish/behaviour_tree/isaac_sim_demo.py --headless
```

To load a different generation file:

```bash
/home/advaith/isaacsim/python.sh tanish/behaviour_tree/isaac_sim_demo.py --json /absolute/path/task_intelligence.JSON
```

If that file requires a starting precondition that the demo does not know, add
it explicitly. Repeat the option for more than one fact:

```bash
/home/advaith/isaacsim/python.sh tanish/behaviour_tree/isaac_sim_demo.py \
  --json /absolute/path/task_intelligence.JSON \
  --initial-fact probe_available
```

If Isaac cannot download its standard Franka asset, provide a local copy:

```bash
/home/advaith/isaacsim/python.sh tanish/behaviour_tree/isaac_sim_demo.py \
  --robot-usd /absolute/path/franka.usd
```

The standard generated primitives supported by this demo are
`navigate_to_workspace`, `perceive_objects`, `grasp_object`, `grasp_tool`,
`manipulate_object`, `trace_linear_path`, `inspect_workspace`, and
`execute_subtask`, plus the demo verifier `verify_block_at_goal`. Motion steps can provide `inputs.position` (`[x, y, z]`) and
`inputs.orientation_wxyz`. Without numeric poses, the demo uses safe fixed poses
to prove the tree-to-simulator connection; production task poses still need to
come from calibration or perception.

## How the integration works

`runtime.py` consumes task-intelligence JSON and runs it one tick per Isaac Sim
frame. It supports `sequence`, `selector`, `parallel`, `retry`, `condition`, and
`action` nodes. A generation file containing only a flat task/subtask list is
still accepted and becomes a Sequence. `isaac_adapters.py` connects each action
node to the existing Franka motion-controller queue.

The generated file describes intent; it does not contain safe robot poses. Each
primitive must therefore be registered with task-specific Isaac code:

```python
from behaviour_tree import BehaviourTreeRuntime, Status, load_task_intelligence
from behaviour_tree.isaac_adapters import controller_primitive, function_primitive

payload = load_task_intelligence("connector_intelligence.JSON")

def queue_grasp(context):
    target = lookup_grasp_pose(context.step.inputs)
    controller.add_cartesian_waypoint(target.position, target.orientation)
    controller.add_gripper_command("close", wait_frames=30)

tree = BehaviourTreeRuntime(
    payload,
    primitives={
        "perceive_objects": function_primitive(run_perception),
        "grasp_object": controller_primitive(queue_grasp, validate=check_grasp),
        "manipulate_object": controller_primitive(queue_insert, validate=check_insert),
    },
    initial_facts={"robot_localized", "workspace_map_loaded", "camera_ready"},
    services={
        "robot": franka,
        "motion_controller": controller,
        "articulation_controller": franka.get_articulation_controller(),
    },
)
```

Tick it from the existing simulation loop only while the timeline is playing:

```python
world.step(render=True)
if world.is_playing():
    status = tree.tick()
    if status is Status.FAILURE:
        print(f"Tree stopped: {tree.feedback}")
```

On Stop/Play reset the scene and call `tree.reset()`. An unknown primitive, a
missing precondition, an empty controller queue, a controller IK failure, or a
failed validator stops the tree with `Status.FAILURE` and a diagnostic message.

Run the host-side tests without Isaac Sim:

```bash
cd Isaacsim-Scripts/tanish
python3 -m unittest discover -s behaviour_tree/tests -v
```

This command tests JSON normalization, ordered execution, selectors, retry,
parallel checks, precondition failure, unknown-primitive failure, and one
controller frame per tree tick. A successful run ends with `Ran 6 tests` and
`OK`.
