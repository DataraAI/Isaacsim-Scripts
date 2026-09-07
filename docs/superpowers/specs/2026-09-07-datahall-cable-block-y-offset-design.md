# DataHall cable and block Y-offset design

## Scope

Modify `/home/aayush/Desktop/Aayush_ws/DataHall_6r.usd` without moving robots,
switches, or unrelated geometry.

## Transform changes

- Apply a `-15` stage-unit Y delta to the three negative-side cable roots and
  their matching `Floor`, `Left`, and `Right` block roots.
- Apply a `+15` stage-unit Y delta to the three positive-side cable roots and
  their matching `Floor`, `Left`, and `Right` block roots.
- Preserve every X/Z coordinate, orientation, and scale.
- Expected Y positions are approximately `-150` on the negative side and `+25`
  on the positive side.

## Method and verification

Create a backup of the USD, edit only each root `xformOp:translate` through the
PXR USD API, save in place, reopen the stage, and verify all 24 affected roots.
