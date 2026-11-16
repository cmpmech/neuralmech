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
- `cartpole_lqr.py`
  linear-quadratic regulator of the linearized cart-pole, bang-bang on the sign of its force; needs no training and sets the reference line of the learning curves
- `cartpole_reinforce.py`
  REINFORCE with discounted returns
- `cartpole_baseline.py`
  REINFORCE with a learned value baseline
- `cartpole_actor_critic.py`
  actor-critic with the temporal-difference error as advantage; policy and value maps
- `cartpole_ppo.py`
  proximal policy optimization with the clipped probability ratio
- `gymnasium_environments.py`
  filmstrips of four frames each of six Gymnasium environments (classic control, Box2D, MuJoCo)
  in a seeded rollout
- `bipedal_walker_ppo.py` -> `models/bipedal_walker_ppo_<iteration>.pt`
  proximal policy optimization with a gaussian policy on the Gymnasium `BipedalWalker-v3`
- `bipedal_walker_filmstrips.py` _needs `bipedal_walker_ppo_<iteration>.pt`_
  filmstrips of the walker untrained, after an intermediate number of iterations, and trained

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

`bipedal_walker_ppo.py` keeps the clipped objective of `cartpole_ppo.py` and adds what a
continuous, long-horizon task needs. The policy is a gaussian whose mean comes from the
network and whose log standard deviation is one free parameter per joint (`model.log_std`,
saved with the model); sampled torques are clipped to $[-1,1]$ only when sent to the
environment, so the log-probabilities stay those of the gaussian. Sixteen copies of the
environment run side by side (`gym.make_vec`) and are reset on the step that ends an
episode (`AutoresetMode.SAME_STEP`); at a truncation after 1600 steps the reward is
increased by $\gamma V$ of the final state, which the environment returns in
`info["final_obs"]`. Advantages use generalized advantage estimation,
$A_\tau=\sum_k(\gamma\lambda)^k\delta_{\tau+k}$ with the temporal-difference errors
$\delta_\tau$, cut at the end of each episode, and are standardized per minibatch. A fall
costs $-100$ against step rewards of order $0.1$, so `NormalizeReward` divides the rewards
by a running standard deviation of the discounted return; this changes the scale but not
the optimal policy. The returns reported are those of the unscaled rewards.
