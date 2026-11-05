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

# environment, steps before the snapshot, crop box (left, top, right, bottom) in pixels
ENVIRONMENTS = [
    ("CartPole-v1", 15, (162, 140, 442, 350)),
    ("Acrobot-v1", 136, (0, 110, 400, 410)),
    ("LunarLander-v3", 45, (140, 100, 460, 340)),
    ("BipedalWalker-v3", 240, (0, 130, 280, 340)),
    ("Ant-v5", 10, None),
    ("Humanoid-v5", 10, None),
]

# mujoco cameras closer than the gymnasium default
CAMERAS = {
    "Ant-v5": dict(trackbodyid=1, distance=3.0, elevation=-30.0, azimuth=90.0),
    "Humanoid-v5": dict(
        trackbodyid=1, distance=2.5, elevation=-15.0, azimuth=160.0,
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
frames = {}
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
    for tau in range(steps):
        s, *_ = env.step(policy(env, s))
    frame = Image.fromarray(env.render())
    if box is not None:
        frame = frame.crop(box).resize(RESOLUTION, Image.LANCZOS)
    frames[name] = frame
    env.close()

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, (name, frame) in zip(axes.flat, frames.items()):
        ax.imshow(frame)
        ax.set_title(name)
        ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, frame in frames.items():
        frame.save(RGB_PNG_DIR / f"gymnasium_{name.split('-')[0].lower()}.png")
