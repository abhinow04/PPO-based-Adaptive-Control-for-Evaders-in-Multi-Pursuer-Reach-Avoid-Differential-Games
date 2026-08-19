import numpy as np

from ppo import deviation_to_velocity


def test_turns_away_from_pursuer():
    evader_pos = np.array([0.0, 0.0])
    target = np.array([10.0, 0.0])
    pursuer = np.array([0.0, 5.0])

    vel = deviation_to_velocity(1.0, evader_pos, target, 1.0, pursuer_positions=[pursuer])
    angle = np.arctan2(vel[1], vel[0])

    # Pursuer is directly above the evader, so the evader should move down
    # (negative y direction) to turn away from the pursuer, not up toward it.
    assert angle < 0.0, f"Expected heading below x-axis, got angle={np.rad2deg(angle):.1f} deg"


if __name__ == "__main__":
    test_turns_away_from_pursuer()
    print("steering test passed")
