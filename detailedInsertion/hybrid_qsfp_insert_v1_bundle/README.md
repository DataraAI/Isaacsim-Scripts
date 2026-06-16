# One-Port Hybrid QSFP Insert v2 Deeper

Changes from v1:
- INSERT_TIP_DEPTH_M changed from 0.048 to 0.060.
- INSERT_STROKE_MAX_FRAMES increased from 1600 to 3200.
- Insert servo now has INSERT_COMMAND_LEAD_LIMIT = 0.014 so the commanded target can get ahead of the measured module and push through contact/friction instead of stalling.
- Pick waypoints use joint_interp=True to avoid the Lula task-space trajectory warnings during pick_hover / pick_grasp / pick_lift.

Run:

```bash
cd /home/aayush/Isaacsim-Scripts
~/isaac-sim-6/python.sh hybrid_qsfp_insert_v2_deeper.py
```
