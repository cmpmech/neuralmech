# Reinforcement Learning Fundamentals

Toy problems for **Chapter 19 (Deep Reinforcement Learning)**. The running example is
the cart-pole of `helper.py` (explicit euler, two push actions, reward 1 per step),
balanced from scratch in PyTorch by five methods; a pendulum swing-up and a 1D gaussian
policy illustrate dynamic programming and the policy gradient.

## Drivers

- `pendulum_value_iteration.py`
  value iteration on the discretized phase plane of a torque-limited pendulum; with `--book`
  also the swing-up of the greedy policy as rod snapshots, split into swings
- `policy_gradient_toy.py`
  score-function gradient of a gaussian policy on a step reward, with and without baseline
- `cartpole_dqn.py`
  deep Q-network with replay buffer and target network
- `cartpole_reinforce.py`
  REINFORCE with discounted returns
- `cartpole_baseline.py`
  REINFORCE with a learned value baseline
- `cartpole_actor_critic.py`
  actor-critic with the temporal-difference error as advantage; policy and value maps
- `cartpole_ppo.py`
  proximal policy optimization with the clipped probability ratio
- `gymnasium_environments.py`
  one frame each of six Gymnasium environments (classic control, Box2D, MuJoCo) after a seeded rollout

## Non-obvious technicalities (authored by Claude)

The cart-pole drivers call `torch.set_num_threads(1)`. With networks of 128 neurons and
one state per forward pass, threading only adds overhead (about $10\times$ slower for the
deep Q-network on a many-core CPU), and the thread count changes the floating-point
rounding, so the run that solves the task in a given number of episodes depends on the
machine otherwise.

Episodes end either because the pole falls (terminal, no value follows) or because the
cap of 500 steps is reached (truncation, the physics continue). The temporal-difference
targets of `cartpole_dqn.py` and `cartpole_actor_critic.py` therefore bootstrap with
$\gamma V(s_{\tau+1})$ at a truncation and drop it only after a fall.
