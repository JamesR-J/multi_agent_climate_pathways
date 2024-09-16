import sys
import jax
import jax.numpy as jnp
from typing import Any
import jax.random as jrandom
from functools import partial
from project_name.agents.PPO_RNN.network import ActorCriticRNN, ScannedRNN  # TODO sort out this class import ting
import optax
from flax.training.train_state import TrainState


class PPO_RNNAgent:
    def __init__(self,
                 env,
                 key,
                 config):
        self.config = config
        self.env = env
        self.network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        init_x = (jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
                  jnp.zeros((1, config["NUM_ENVS"])),
                  )
        key, _key = jrandom.split(key)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        self.network_params = self.network.init(_key, init_hstate, init_x)
        self.init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"],
                                                       config["GRU_HIDDEN_DIM"])  # TODO do we need both?

        def linear_schedule(count):  # TODO put this somewhere better and think this is right?
            if config["SPLIT_TRAIN"]:
                count += config["NUM_UPDATES"]
            frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["TOTAL_UPDATES"])
            # frac = 1 - count // 16 / num_updates
            return config["LR"] * frac

        if config["ANNEAL_LR"]:
            self.tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                                  optax.adam(learning_rate=linear_schedule, eps=1e-5),
                                  )
        else:
            self.tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                                  optax.adam(config["LR"], eps=1e-5),
                                  )

    def create_train_state(self):
        return (TrainState.create(apply_fn=self.network.apply,
                                  params=self.network_params,
                                  tx=self.tx),
                self.init_hstate,
                )

    @partial(jax.jit, static_argnums=(0))
    def act(self, train_state: Any, hstate: Any, ac_in: Any, key: Any):  # TODO better implement checks
        hstate, pi, value, action_logits = train_state.apply_fn(train_state.params, hstate, ac_in)
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        return hstate, action, log_prob, value, key, action_logits, _key  # TODO do we need to return key?

    @partial(jax.jit, static_argnums=(0))
    def update(self, runner_state, traj_batch):
        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, last_done, hstate, key = runner_state
        # avail_actions = jnp.ones(self.env.action_space(self.env.agents[0]).n)
        ac_in = (last_obs[jnp.newaxis, :],
                 last_done[jnp.newaxis, :],
                 # avail_actions[jnp.newaxis, :],
                 )
        _, _, last_val, _ = train_state.apply_fn(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze()

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + self.config["GAMMA"] * next_value * (1 - done) - value
                gae = (delta + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - done) * gae)
                return (gae, value), gae

            _, advantages = jax.lax.scan(_get_advantages,
                                         (jnp.zeros_like(last_val), last_val),
                                         traj_batch,
                                         reverse=True,
                                         unroll=16,
                                         )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    # RERUN NETWORK
                    _, pi, value, _ = train_state.apply_fn(params,
                                                      init_hstate.squeeze(axis=0),
                                                      (traj_batch.obs,
                                                       traj_batch.done,
                                                       # traj_batch.avail_actions
                                                       ),
                                                      )
                    sys.exit()
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-self.config["CLIP_EPS"],
                                                                                            self.config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(
                        where=(1 - traj_batch.done))

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (jnp.clip(ratio,
                                            1.0 - self.config["CLIP_EPS"],
                                            1.0 + self.config["CLIP_EPS"],
                                            ) * gae)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
                    entropy = pi.entropy().mean(where=(1 - traj_batch.done))

                    total_loss = (loss_actor
                                  + self.config["VF_COEF"] * value_loss
                                  - self.config["ENT_COEF"] * entropy
                                  )

                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, init_hstate, traj_batch, advantages, targets, key = update_state
            key, _key = jrandom.split(key)

            # adding an additional "fake" dimensionality to perform minibatching correctly
            init_hstate = jnp.reshape(init_hstate, (1, self.config["NUM_ENVS"], -1))

            permutation = jrandom.permutation(_key, self.config["NUM_ENVS"])
            print(traj_batch)
            traj_batch = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
            batch = (init_hstate,  # TODO check this axis swapping etc if it works
                     traj_batch,
                     jnp.swapaxes(advantages, 0, 1).squeeze(),
                     jnp.swapaxes(targets, 0, 1).squeeze())
            print(batch)
            print("2")
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
            print(shuffled_batch)
            print("3")

            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(
                jnp.reshape(x, [x.shape[0], self.config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]), ), 1, 0, ),
                                                 shuffled_batch, )
            print(minibatches)
            # sys.exit()
            train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)

            traj_batch = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)  # TODO dodge to swap back again

            update_state = (train_state,
                            init_hstate.squeeze(),
                            traj_batch,
                            advantages,
                            targets,
                            key,
                            )
            return update_state, total_loss

        update_state = (train_state, hstate, traj_batch, advantages, targets, key)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, self.config["UPDATE_EPOCHS"])
        train_state, hstate, traj_batch, advantages, targets, key = update_state

        return train_state, env_state, last_obs, last_done, hstate, key
