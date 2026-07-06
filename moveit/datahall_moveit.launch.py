from pathlib import Path

from ament_index_python.packages import get_package_share_directory


ROBOT_BASE_TRANSLATION = ("0.0", "0.0", "0.0")


def _patch_official_launch_source(source: str) -> str:
    """Patch the installed Isaac MoveIt launch source without editing the workspace."""
    patched = source

    replacements = {
        '"0", "-0.64", "0"': f'"{ROBOT_BASE_TRANSLATION[0]}", "{ROBOT_BASE_TRANSLATION[1]}", "{ROBOT_BASE_TRANSLATION[2]}"',
        "'0', '-0.64', '0'": f"'{ROBOT_BASE_TRANSLATION[0]}', '{ROBOT_BASE_TRANSLATION[1]}', '{ROBOT_BASE_TRANSLATION[2]}'",
        '"0.0", "-0.64", "0.0"': f'"{ROBOT_BASE_TRANSLATION[0]}", "{ROBOT_BASE_TRANSLATION[1]}", "{ROBOT_BASE_TRANSLATION[2]}"',
        "'0.0', '-0.64', '0.0'": f"'{ROBOT_BASE_TRANSLATION[0]}', '{ROBOT_BASE_TRANSLATION[1]}', '{ROBOT_BASE_TRANSLATION[2]}'",
        '"-4.7", "-6.1", "0.8"': f'"{ROBOT_BASE_TRANSLATION[0]}", "{ROBOT_BASE_TRANSLATION[1]}", "{ROBOT_BASE_TRANSLATION[2]}"',
        "'-4.7', '-6.1', '0.8'": f"'{ROBOT_BASE_TRANSLATION[0]}', '{ROBOT_BASE_TRANSLATION[1]}', '{ROBOT_BASE_TRANSLATION[2]}'",
    }

    for old, new in replacements.items():
        patched = patched.replace(old, new)

    if patched == source:
        print(
            "[datahall_moveit] WARNING: did not find the expected static TF "
            "translation in isaac_moveit.launch.py. The launch will still run, "
            "but world -> panda_link0 may use the official sample offset."
        )
    else:
        print(
            "[datahall_moveit] Patched world -> panda_link0 translation to "
            f"{' '.join(ROBOT_BASE_TRANSLATION)}"
        )

    return patched


def generate_launch_description():
    isaac_moveit_share = Path(get_package_share_directory("isaac_moveit"))
    official_launch = isaac_moveit_share / "launch" / "isaac_moveit.launch.py"

    if not official_launch.exists():
        raise FileNotFoundError(f"Could not find official launch file: {official_launch}")

    source = official_launch.read_text()
    patched_source = _patch_official_launch_source(source)

    globals_for_launch = {
        "__file__": str(official_launch),
        "__name__": "datahall_patched_isaac_moveit_launch",
    }
    exec(compile(patched_source, str(official_launch), "exec"), globals_for_launch)

    if "generate_launch_description" not in globals_for_launch:
        raise RuntimeError(f"{official_launch} does not define generate_launch_description()")

    return globals_for_launch["generate_launch_description"]()
