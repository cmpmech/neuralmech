import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from PIL import Image

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

# -------------------------------------- settings -------------------------------------
SEED = 0
STAGES = {"untrained": 0, "intermediate": 75, "final": 842}  # training iterations
FRAME_STEPS = [0, 20, 40, 60, 80, 100]  # steps of the frames
BOX = (20, 150, 260, 330)  # crop box (left, top, right, bottom) in pixels
RESOLUTION = (480, 360)  # width and height of every frame in pixels

# -------------------------------------- rollout --------------------------------------
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
filmstrips = {}
for stage, iteration in STAGES.items():
    model = torch.load(
        MODEL_DIR / f"bipedal_walker_ppo_{iteration}.pt", weights_only=False,
        map_location=device,
    )
    s, _ = env.reset(seed=SEED)
    frames = [env.render()]
    episode_return = 0
    done = False
    with torch.no_grad():
        while not done:
            mean = model(torch.as_tensor(s, dtype=torch.float32).to(device))
            action = torch.distributions.Normal(mean, model.log_std.exp()).sample()
            action = action.clamp(-1, 1).cpu().numpy()
            s, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            episode_return += reward
            done = terminated or truncated
    print(f"{stage}: return {episode_return:.1f} after {len(frames) - 1} steps")
    frames = [frames[min(tau, len(frames) - 1)] for tau in FRAME_STEPS]
    filmstrips[stage] = [
        Image.fromarray(frame).crop(BOX).resize(RESOLUTION, Image.LANCZOS)
        for frame in frames
    ]
env.close()

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(len(STAGES), len(FRAME_STEPS), figsize=(9, 4))
    for row, (stage, filmstrip) in zip(axes, filmstrips.items()):
        for ax, frame in zip(row, filmstrip):
            ax.imshow(frame)
            ax.axis("off")
        row[0].set_title(stage, loc="left")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for stage, filmstrip in filmstrips.items():
        for i, frame in enumerate(filmstrip):
            frame.save(RGB_PNG_DIR / f"bipedal_walker_{stage}_{i}.png")
