import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from ..infrastructure import pytorch_util as ptu
from ..policies.base_policy import BasePolicy



class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray,deter = False) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        ditrib = self.forward(ptu.from_numpy(observation), deter)
        with torch.no_grad():
            if not self.discrete:
                if not deter:
                    action = ptu.to_numpy(ditrib.rsample())
                else:
                    action = ptu.to_numpy(ditrib)
            else:
                action = ptu.to_numpy(torch.argmax(ditrib))
        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    def forward(self, observation: torch.FloatTensor,deter=False) -> Any:
        if self.discrete:
            distrib = self.logits_na(observation)
            return distrib
        else:
            std = torch.exp(self.logstd)
            mu = self.mean_net(observation)
            if deter:
                return mu
            else:
                return Normal(mu,std)



class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        assert self.training, 'Policy must be created with training=True in order to perform training updates...'

        # TODO: update the policy and return the loss
        policy_act = self.forward(ptu.from_numpy(observations)).rsample()
        loss = self.loss(policy_act,ptu.from_numpy(actions))
        # loss = (policy_act - ptu.from_numpy(actions)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
