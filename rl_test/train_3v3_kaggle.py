# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
# Modified for Kaggle/Colab with auto-save and resume support

import argparse
import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import time
from pyquaticus.envs.pyquaticus import Team
from pyquaticus import pyquaticus_v0
from ray.rllib.policy.policy import Policy
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std
import logging
import shutil

class RandPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self, obs_batch, state_batches, prev_action_batch=None,
                        prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3v3 on Kaggle/Colab with auto-save')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint', default=None)
    parser.add_argument('--start-iter', type=int, help='Starting iteration', default=0)
    parser.add_argument('--max-iter', type=int, help='Max iterations', default=8001)
    parser.add_argument('--save-interval', type=int, help='Save every N iterations', default=100)
    parser.add_argument('--backup-dir', type=str, help='Backup directory (e.g., Google Drive)', default=None)
    
    reward_config = {'agent_0':rew.caps_and_grabs, 'agent_1':rew.caps_and_grabs, 
                     'agent_2':rew.caps_and_grabs, 'agent_3':None, 'agent_4':None, 'agent_5':None}
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict, render_mode=None, reward_config=reward_config, team_size=3)
    env = ParallelPettingZooWrapper(env_creator(config_dict))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    
    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 'agent_0':
            return "agent-0-policy"
        if agent_id == 'agent_1':
            return "agent-1-policy"
        if agent_id == 'agent_2':
            return "agent-2-policy"
        return "random"
    
    policies = {
        'agent-0-policy': (None, obs_space, act_space, {}), 
        'agent-1-policy': (None, obs_space, act_space, {}),
        'agent-2-policy': (None, obs_space, act_space, {}),
        'random': (RandPolicy, obs_space, act_space, {"no_checkpoint": True})
    }
    
    env.close()
    
    # Config with GPU support and increased workers
    ppo_config = (PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env='pyquaticus')
        .framework("torch")
        .resources(num_gpus=1)  # Use GPU if available
        .env_runners(num_env_runners=2, num_cpus_per_env_runner=1, sample_timeout_s=600.0)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, 
                     policies_to_train=["agent-0-policy", "agent-1-policy", "agent-2-policy"])
    )
    
    # Resume or build new
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from: {args.resume}")
        algo = ppo_config.build_algo()
        algo.restore(args.resume)
        start_iter = args.start_iter
    else:
        print("🆕 Starting fresh training")
        algo = ppo_config.build_algo()
        start_iter = 0
    
    print(f"🚀 Training from iteration {start_iter} to {args.max_iter}")
    print(f"💾 Saving every {args.save_interval} iterations")
    
    for i in range(start_iter, args.max_iter):
        print(f"\n{'='*50}")
        print(f"Iteration {i}/{args.max_iter}")
        start_time = time.time()
        
        result = algo.train()
        
        elapsed = time.time() - start_time
        print(f"⏱️ Time: {elapsed:.1f}s | Reward: {result.get('env_runners', {}).get('episode_reward_mean', 0):.2f}")
        
        # Save checkpoint
        if i % args.save_interval == 0:
            chkpt_path = f'./ray_test/iter_{i}/'
            print(f"💾 Saving checkpoint to {chkpt_path}")
            algo.save(chkpt_path)
            
            # Backup to external drive (Google Drive) if specified
            if args.backup_dir:
                backup_path = os.path.join(args.backup_dir, f'iter_{i}')
                print(f"☁️ Backing up to {backup_path}")
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                shutil.copytree(chkpt_path, backup_path)
            
            print(f"✅ Checkpoint saved!")
    
    print("\n🎉 Training completed!")
    final_path = './ray_test/iter_final/'
    algo.save(final_path)
    print(f"💾 Final model saved to {final_path}")
