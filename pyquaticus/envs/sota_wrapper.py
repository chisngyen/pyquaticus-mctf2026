"""
SOTA Environment Wrapper for PyQuaticus.

Red team (agents 3,4,5) is controlled by heuristic bots internally.
RLlib only sees Blue team (agents 0,1,2).
This avoids the info_batch issue where RLlib doesn't pass env info to policies.
"""

import functools
import numpy as np
from pettingzoo import ParallelEnv

from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent


class SotaEnvWrapper(ParallelEnv):
    """PyQuaticus wrapper: Red team controlled by heuristic bots internally."""

    metadata = {"render_modes": ["human"], "name": "pyquaticus_sota"}

    def __init__(self, config_dict, reward_config, team_size=3,
                 render_mode=None, opponent_mode='hard'):
        super().__init__()

        self.base_env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict,
            render_mode=render_mode,
            reward_config=reward_config,
            team_size=team_size
        )

        self.opponent_mode = opponent_mode
        self.blue_agents = ['agent_0', 'agent_1', 'agent_2']
        self.red_agents = ['agent_3', 'agent_4', 'agent_5']

        self.possible_agents = list(self.blue_agents)
        self.agents = list(self.blue_agents)

        # Heuristic policies for Red team
        self.red_policies = {
            'agent_3': BaseAttacker('agent_3', self.base_env, mode=opponent_mode),
            'agent_4': BaseDefender('agent_4', self.base_env, mode=opponent_mode),
            'agent_5': Heuristic_CTF_Agent('agent_5', self.base_env, mode=opponent_mode),
        }

        self._all_obs = {}
        self._all_infos = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.base_env.observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.base_env.action_space(agent)

    def reset(self, seed=None, options=None):
        obs, infos = self.base_env.reset(seed=seed, options=options)
        self._all_obs = obs
        self._all_infos = infos
        self.agents = [a for a in self.blue_agents if a in obs]
        blue_obs = {k: v for k, v in obs.items() if k in self.blue_agents}
        blue_infos = {k: v for k, v in infos.items() if k in self.blue_agents}
        return blue_obs, blue_infos

    def step(self, blue_actions):
        # Compute red team actions using heuristics
        red_actions = {}
        for agent_id in self.red_agents:
            if agent_id in self._all_obs:
                obs = self._all_obs[agent_id]
                info = self._all_infos.get(agent_id, {})
                try:
                    red_actions[agent_id] = self.red_policies[agent_id].compute_action(obs, info)
                except Exception:
                    red_actions[agent_id] = self.base_env.action_space(agent_id).sample()

        all_actions = {**blue_actions, **red_actions}
        obs, rewards, terminateds, truncateds, infos = self.base_env.step(all_actions)

        self._all_obs = obs
        self._all_infos = infos

        self.agents = [a for a in self.blue_agents if a in obs]
        blue_obs = {k: v for k, v in obs.items() if k in self.blue_agents}
        blue_rewards = {k: v for k, v in rewards.items() if k in self.blue_agents}
        blue_terms = {k: v for k, v in terminateds.items() if k in self.blue_agents}
        blue_truncs = {k: v for k, v in truncateds.items() if k in self.blue_agents}
        blue_infos = {k: v for k, v in infos.items() if k in self.blue_agents}
        return blue_obs, blue_rewards, blue_terms, blue_truncs, blue_infos

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
