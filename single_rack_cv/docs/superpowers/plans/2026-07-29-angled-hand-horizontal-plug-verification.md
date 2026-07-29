# Corrected Verification

Pull branch head, run the three focused test modules, then launch `main.py`. The expected startup geometry is:

- preserved plug-tip target `[0.7666, -0.1375, 1.3]`
- solved hand target approximately `[0.882128, -0.1375, 1.3667]`
- measured pitch within 30 +/- 0.5 degrees
- palm-side error <= 1 degree
- plug horizontal error <= 1 degree
- requested pitch sign valid

Do not proceed to perception or insertion if any geometry gate fails.
