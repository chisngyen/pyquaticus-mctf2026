
import argparse
import gymnasium as gym
import numpy as np
import ray
import time
import os
import shutil
import logging
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policy_wrappers import (
    AttackGen, DefendGen, CombinedGen, RandPolicy
)
from pyquaticus.envs.pyquaticus import Team

def main():
    parser = argparse.ArgumentParser(description='Train 3v3 SOTA Agent (Local)')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--max-iter', type=int, default=8000, help='Max iterations')
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)

    # 1. Environment Configuration
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True
    
    reward_config = {
        'agent_0': rew.caps_and_grabs,
        'agent_1': rew.caps_and_grabs,
        'agent_2': rew.caps_and_grabs,
        'agent_3': None,
        'agent_4': None,
        'agent_5': None
    }

    render_mode = 'human' if args.render else None

    # 2. Register Environment
    def env_creator(config):
        return pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict,
            render_mode=render_mode,
            reward_config=reward_config,
            team_size=3
        )

    register_env('pyquaticus_sota', lambda config: ParallelPettingZooWrapper(env_creator(config)))

    # 3. Define Diverse Heuristic Opponents
    dummy_env = ParallelPettingZooWrapper(env_creator(config_dict))
    obs_space = dummy_env.observation_space['agent_0']
    act_space = dummy_env.action_space['agent_0']
    
    # Red team: 1 Attacker + 1 Defender + 1 Combined (all 'hard' mode)
    attack_policy_cls = AttackGen("agent_3", dummy_env.par_env, mode='hard')
    defend_policy_cls = DefendGen("agent_4", dummy_env.par_env, mode='hard')
    combined_policy_cls = CombinedGen("agent_5", dummy_env.par_env, mode='hard')
    
    dummy_env.close()

    policies = {
        'learned_policy': (None, obs_space, act_space, {}),
        'attack_opponent': (attack_policy_cls, obs_space, act_space, {"no_checkpoint": True}),
        'defend_opponent': (defend_policy_cls, obs_space, act_space, {"no_checkpoint": True}),
        'combined_opponent': (combined_policy_cls, obs_space, act_space, {"no_checkpoint": True}),
    }

    # 4. Policy Mapping
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in ['agent_0', 'agent_1', 'agent_2']:
            return "learned_policy"
        elif agent_id == 'agent_3':
            return "attack_opponent"
        elif agent_id == 'agent_4':
            return "defend_opponent"
        else:
            return "combined_opponent"

    # 5. PPO Configuration (GPU)
    ppo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .environment(env='pyquaticus_sota')
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(num_env_runners=2, num_cpus_per_env_runner=1,
                     sample_timeout_s=600.0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["learned_policy"]
        )
    )

    # 6. Build Algorithm
    if args.resume:
        print(f"📂 Resuming from {args.resume}...")
        algo = ppo_config.build_algo()
        algo.restore(args.resume)
    else:
        print("🆕 Starting Fresh SOTA Training")
        print("   🔴 Red: Attacker(hard) + Defender(hard) + Combined(hard)")
        print("   🔵 Blue: PPO (shared policy, GPU)")
        algo = ppo_config.build_algo()

    # 7. Training Loop
    save_dir = "./mctf_checkpoints_sota/"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(args.max_iter):
        start_t = time.time()
        result = algo.train()
        elapsed = time.time() - start_t
        
        er = result.get('env_runners', {})
        reward_mean = er.get('episode_reward_mean', 0)
        print(f"Iter {i:>5}: Reward={reward_mean:>8.2f} | Time={elapsed:.1f}s")

        if i % 50 == 0:
            algo.save(save_dir + f"iter_{i}")
            print(f"   💾 Checkpoint saved!")

if __name__ == "__main__":
    main()
