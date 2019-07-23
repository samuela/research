from typing import Any, NamedTuple, Callable, TypeVar

from jax import lax, grad, ops, random, tree_util, vmap
import jax.numpy as jp

from research.statistax import Distribution
from research.utils import Optimizer

State = TypeVar("State")
Action = TypeVar("Action")

class Env(NamedTuple):
  initial_distribution: Distribution
  step: Callable[[State, Action], Distribution]
  reward: Callable[[State, Action, State], jp.ndarray]

def rollout(rng, env: Env, policy, num_timesteps: int):
  init_rng, steps_rng = random.split(rng)
  init_state = env.initial_distribution.sample(init_rng)
  return rollout_from_state(steps_rng, env, policy, num_timesteps, init_state)

def rollout_from_state(rng, env: Env, policy, num_timesteps: int, state):
  def step(state, step_rng):
    action_rng, dynamics_rng = random.split(step_rng)
    action = policy(state).sample(action_rng)
    next_state = env.step(state, action).sample(dynamics_rng)
    return next_state, (state, action)

  _, res = lax.scan(step, state, random.split(rng, num_timesteps))
  return res

class ReplayBuffer(NamedTuple):
  states: jp.ndarray
  actions: jp.ndarray
  rewards: jp.ndarray
  next_states: jp.ndarray
  count: int

  @property
  def buffer_size(self):
    return self.states.shape[0]

  def add(self, state, action, reward, next_state):
    ix = self.count % self.buffer_size
    return ReplayBuffer(
        states=ops.index_update(self.states, ix, state),
        actions=ops.index_update(self.actions, ix, action),
        rewards=ops.index_update(self.rewards, ix, reward),
        next_states=ops.index_update(self.next_states, ix, next_state),
        count=self.count + 1,
    )

  def minibatch(self, rng, batch_size: int):
    ixs = random.randint(
        rng,
        (batch_size, ),
        minval=0,
        maxval=jp.minimum(self.buffer_size, self.count),
    )
    return (
        self.states[ixs, ...],
        self.actions[ixs, ...],
        self.rewards[ixs, ...],
        self.next_states[ixs, ...],
    )

def ddpg_step(
    rng,
    params,
    tracking_params,
    env: Env,
    gamma: float,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    actor,
    critic,
    state,
    noise: Distribution,
):
  """Calculate the gradient from a single step of DDPG.

  Args:
    rng: The PRNG key.
    params (tuple): The current (actor, critic) parameters.
    tracking_params (tuple): The tracking (actor, critic) parameters.
    env (Env): The environment to operate in.
    gamma (float): The time discount factor.
    replay_buffer (ReplayBuffer): The experience replay buffer.
    batch_size (int): The size of the batch to pull out of `replay_buffer`.
    actor: The actor function, `\\mu(s)`.
    critic: The critic function, `Q(s, a)`.
    state: The current state.
    noise: A function for the action noise distribution at each time step.
  """
  actor_params, critic_params = params
  tracking_actor_params, tracking_critic_params = tracking_params
  Q_track = lambda s, a: critic(tracking_critic_params, (s, a))
  mu_track = lambda s: actor(tracking_actor_params, s)

  rng_noise, rng_transition, rng_minibatch = random.split(rng, 3)

  actor_action = actor(actor_params, state)
  action_noise = noise.sample(rng_noise)

  # We corrupt the actor_action with noise in order to promote exploration.
  action = actor_action + action_noise
  next_state = env.step(state, action).sample(rng_transition)
  reward = env.reward(state, action, next_state)
  new_rb = replay_buffer.add(state, action, reward, next_state)

  # Sample minibatch from the replay buffer.
  replay_states, replay_actions, replay_rewards, replay_next_states = new_rb.minibatch(
      rng_minibatch, batch_size)

  # When ns is terminal, the "gamma * Q_track" term should be dropped.
  replay_ys = vmap(lambda r, ns: r + gamma * Q_track(ns, mu_track(ns)),
                   in_axes=(0, 0))(replay_rewards, replay_next_states)

  def critic_loss(p):
    replay_pred_ys = vmap(lambda s, a: critic(p, (s, a)),
                          in_axes=(0, 0))(replay_states, replay_actions)
    return jp.mean((replay_ys - replay_pred_ys)**2.0)

  critic_grad = grad(critic_loss)(critic_params)

  def actor_loss(p):
    # Easier to represent it this way instead of expanding the chain rule, as is
    # done in the paper.
    loss_single = lambda s: -critic(critic_params, (s, actor(p, s)))
    return jp.mean(vmap(loss_single)(replay_states))

  actor_grad = grad(actor_loss)(actor_params)

  # Note: There's potentially a slight deviation from the paper here in the
  # sense that it says "update the critic, then update the actor" but the
  # gradient of the actor depends on the critic. For simplicity, this
  # implementation calculates the gradient of the actor without updating the
  # critic first. This should have a negligible impact on behavior.
  return (actor_grad, critic_grad), reward, next_state, new_rb, action_noise

class LoopState(NamedTuple):
  optimizer: Optimizer
  tracking_params: Any
  cumulative_reward: jp.ndarray
  state: State
  replay_buffer: ReplayBuffer
  prev_noise: jp.ndarray

def ddpg_episode(
    env: Env,
    gamma: float,
    tau: float,
    actor,
    critic,
    noise: Callable[[int, jp.ndarray], Distribution],
    epside_length: int,
    batch_size: int,
):
  """Run DDPG for a single episode.

  Args:
    env (Env): The environment to run the agent in.
    gamma (float): The time discount factor.
    tau (float): The parameter tracking rate.
    actor: The actor function, `\\mu(s)`.
    critic: The critic function, `Q(s, a)`.
    noise: A function for the action noise distribution at each time step.
    episode_length (int): The length of the episode to run.
    batch_size (int): The size of batches to be pulled out of the replay buffer.

  Returns:
    run: A `jit`-able function to actually run the episode.
  """

  def run(
      rng,
      init_noise: Distribution,
      init_replay_buffer: ReplayBuffer,
      init_optimizer: Optimizer,
      init_tracking_params,
  ) -> LoopState:
    """A curried `jit`-able function to actually run the DDPG episode."""
    rng_start, rng_init_noise, rng_rest = random.split(rng, 3)
    rngs = random.split(rng_rest, epside_length)

    def step(i, loop_state: LoopState):
      g, reward, next_state, new_replay_buffer, new_prev_noise = ddpg_step(
          rngs[i],
          loop_state.optimizer.value,
          loop_state.tracking_params,
          env,
          gamma,
          loop_state.replay_buffer,
          batch_size,
          actor,
          critic,
          loop_state.state,
          noise(i, loop_state.prev_noise),
      )
      new_cumulative_reward = loop_state.cumulative_reward + (gamma**
                                                              i) * reward
      new_optimizer = loop_state.optimizer.update(g)
      new_tracking_params = tree_util.tree_multimap(
          lambda new, old: tau * new + (1 - tau) * old,
          new_optimizer.value,
          loop_state.tracking_params,
      )
      return LoopState(
          new_optimizer,
          new_tracking_params,
          new_cumulative_reward,
          next_state,
          new_replay_buffer,
          new_prev_noise,
      )

    init_val = LoopState(
        optimizer=init_optimizer,
        tracking_params=init_tracking_params,
        cumulative_reward=jp.array(0.0),
        state=env.initial_distribution.sample(rng_start),
        replay_buffer=init_replay_buffer,
        prev_noise=init_noise.sample(rng_init_noise),
    )

    return lax.fori_loop(0, epside_length, step, init_val)

  return run
