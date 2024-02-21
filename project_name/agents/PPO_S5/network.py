import sys

import flax.linen as nn
import functools
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import distrax
from .S5 import init_S5SSM, make_DPLR_HiPPO, StackedEncoderModel


class ActorCriticS5(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    ssm_init_fn: Any

    def setup(self):
        self.encoder_0 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.encoder_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.action_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_decoder = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

        self.value_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_decoder = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.s5 = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.config["S5_D_MODEL"],
            n_layers=self.config["S5_N_LAYERS"],
            activation=self.config["S5_ACTIVATION"],
            do_norm=self.config["S5_DO_NORM"],
            prenorm=self.config["S5_PRENORM"],
            do_gtrxl_norm=self.config["S5_DO_GTRXL_NORM"],
        )
        if self.config["CONTINUOUS"]:
            self.log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, hidden, x):
        obs, dones = x
        if self.config.get("NO_RESET"):
            dones = jnp.zeros_like(dones)
        embedding = self.encoder_0(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = self.encoder_1(embedding)
        embedding = nn.leaky_relu(embedding)

        hidden, embedding = self.s5(hidden, embedding, dones)

        actor_mean = self.action_body_0(embedding)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_body_1(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_decoder(actor_mean)

        if self.config["CONTINUOUS"]:
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = self.value_body_0(embedding)
        critic = nn.leaky_relu(critic)
        critic = self.value_body_1(critic)
        critic = nn.leaky_relu(critic)
        critic = self.value_decoder(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
