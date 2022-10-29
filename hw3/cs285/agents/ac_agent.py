from collections import OrderedDict

from ..critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ..infrastructure.replay_buffer import ReplayBuffer
from ..infrastructure.utils import *
from ..policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        policy_loss = []
        critic_loss = []

        for iter in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss.append(self.critic.update(ob_no,ac_na,next_ob_no,re_n,terminal_n))

        advantages = self.estimate_advantage(ob_no,next_ob_no,re_n,terminal_n)

        for iter in range(self.agent_params['num_actor_updates_per_agent_update']):
            policy_loss.append(self.actor.update(ob_no,ac_na,advantages))

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss[-1]
        loss['Actor_Loss'] =  policy_loss[-1]

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:

        value = self.critic.forward_np(ob_no)
        next_value = self.critic.forward_np(next_ob_no)
        adv_n = re_n + self.gamma * next_value * (1 - terminal_n) - value

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
