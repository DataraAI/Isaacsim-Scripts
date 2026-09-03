# Cable support grasp notch Design

**Date:** 2026-09-02  
**Status:** Approved  
**Location:** `aayush/asset_spawn/spawn.py`

## Goal

Cut a finger-clearance notch in `/World/CableSupportBlock` under
`E_part006_44` so a Robotiq 2F-85 can pinch both sides of crystal head 45
without colliding with the block top. Keep the remainder of the block surface
so the cable body / head39 end stay supported.

## Non-goals

- Changing cable seating Z or robot motion
- Enabling cable physics

## Approach

After crystal heads are seated, rebuild `/World/CableSupportBlock` as a **U-channel**:
`Floor` + `Left` + `Right` under `E_part006_44`.

- Notch width = part X span + `NOTCH_EXTRA_X_M` (**0.030 m**)
- Notch depth from top = `NOTCH_DEPTH_Z_M` (**0.045 m**); floor remains for balance
- If the part sits at a block end, extend the block by `NOTCH_END_SHOULDER_M` so
  both walls stay visible (a through-cut at the tip only looked like a shorter block)

## Tunables

| Name | Default | Meaning |
|------|---------|---------|
| `NOTCH_EXTRA_X_M` | `0.030` | Extra X clearance beyond part bbox |
| `NOTCH_DEPTH_Z_M` | `0.045` | Channel depth from block top |
| `NOTCH_END_SHOULDER_M` | `0.028` | Min wall past notch at block ends |
| `NOTCH_MIN_SEGMENT_X_M` | `0.012` | Min Left/Right wall width |

## Success

- Visible gap under `E_part006_44`
- Solid block surface under the rest of the cable
- Both pieces have collision enabled
