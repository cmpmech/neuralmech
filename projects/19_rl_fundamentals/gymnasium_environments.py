import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics
from gymnasium.envs.box2d.lunar_lander import heuristic
from PIL import Image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()

os.environ.setdefault("MUJOCO_GL", "egl")  # headless mujoco rendering, or osmesa

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
SEED = 0
RESOLUTION = (480, 360)  # width and height of every frame in pixels

# environment, steps of the frames, crop box (left, top, right, bottom) in pixels
ENVIRONMENTS = [
    ("CartPole-v1", [3, 6, 9, 12, 15, 18], (162, 140, 442, 350)),
    ("Acrobot-v1", [206, 210, 214, 218, 222, 226], (17, 120, 483, 470)),
    ("LunarLander-v3", [20, 44, 68, 92, 116, 140], (140, 60, 460, 300)),
    ("BipedalWalker-v3", [0, 60, 120, 180, 240, 300], (0, 130, 280, 340)),
    ("Ant-v5", [0, 4, 8, 12, 16, 20], None),
    ("Humanoid-v5", [0, 5, 10, 15, 20, 25], None),
]

# mujoco cameras closer than the gymnasium default
CAMERAS = {
    "Ant-v5": dict(trackbodyid=1, distance=3.0, elevation=-30.0, azimuth=90.0),
    "Humanoid-v5": dict(
        trackbodyid=1, distance=2.0, elevation=-15.0, azimuth=160.0,
        lookat=np.array([0.0, 0.0, 1.0]),
    ),
}

# --------------------------------------- helper --------------------------------------
walker = BipedalWalkerHeuristics()
POLICIES = {
    "Acrobot-v1": lambda env, s: 2 if s[4] > 0 else 0,  # torque with swing pumps energy
    "LunarLander-v3": heuristic,
    "BipedalWalker-v3": lambda env, s: walker.step_heuristic(s),
}
random_policy = lambda env, s: env.action_space.sample()

# -------------------------------------- rollout --------------------------------------
filmstrips = {}
for name, steps, box in ENVIRONMENTS:
    if name in CAMERAS:
        env = gym.make(
            name, render_mode="rgb_array", width=RESOLUTION[0], height=RESOLUTION[1],
            default_camera_config=CAMERAS[name],
        )
    else:
        env = gym.make(name, render_mode="rgb_array")
    policy = POLICIES.get(name, random_policy)
    s, _ = env.reset(seed=SEED)
    env.action_space.seed(SEED)
    frames = [env.render()]
    for tau in range(steps[-1]):
        s, *_ = env.step(policy(env, s))
        frames.append(env.render())
    filmstrips[name] = []
    for tau in steps:
        frame = Image.fromarray(frames[tau])
        if box is not None:
            frame = frame.crop(box).resize(RESOLUTION, Image.LANCZOS)
        filmstrips[name].append(frame)
    env.close()

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(len(ENVIRONMENTS), 6, figsize=(12, 9))
    for row, (name, filmstrip) in zip(axes, filmstrips.items()):
        for ax, frame in zip(row, filmstrip):
            ax.imshow(frame)
            ax.axis("off")
        row[0].set_title(name, loc="left")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, filmstrip in filmstrips.items():
        for i, frame in enumerate(filmstrip):
            frame.save(RGB_PNG_DIR / f"gymnasium_{name.split('-')[0].lower()}_{i}.png")
