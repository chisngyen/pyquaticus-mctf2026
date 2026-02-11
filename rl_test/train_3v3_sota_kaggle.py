#!/usr/bin/env python3
"""
MCTF 2026 - SOTA Training Script (Kaggle H100)
============================================
Usage on Kaggle notebook:

    Cell 1:
        !git clone https://github.com/chisngyen/pyquaticus-mctf2026.git /kaggle/working/pyquaticus
        %cd /kaggle/working/pyquaticus
        !pip install -q pettingzoo pygame shapely scipy lz4
        !python rl_test/train_3v3_sota_kaggle.py --max-iter 2000

    Cell 2 (resume after timeout):
        %cd /kaggle/working/pyquaticus
        !python rl_test/train_3v3_sota_kaggle.py --resume latest --max-iter 4000
"""

import sys, os, warnings, importlib, argparse, time, shutil, logging
import functools
import enum

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# 1. AUTO-SETUP: paths + gymnasium patch
# ============================================================================
# Add repo root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Patch gymnasium on disk (Ray workers need this)
import gymnasium
import gymnasium.envs.registration as _reg

def _patch_file(filepath, label):
    with open(filepath, 'r') as f:
        content = f.read()
    if 'VectorizeMode' not in content:
        with open(filepath, 'a') as f:
            f.write('''
import enum as _patch_enum
class VectorizeMode(_patch_enum.Enum):
    SYNC = "sync"
    ASYNC = "async"
    AUTORESET = "autoreset"
''')
        print(f"  🔧 Patched {label}")
        return True
    return False

_gym_init = os.path.join(os.path.dirname(gymnasium.__file__), '__init__.py')
_reg_file = os.path.join(os.path.dirname(_reg.__file__), 'registration.py')

patched = _patch_file(_gym_init, 'gymnasium/__init__.py')
patched |= _patch_file(_reg_file, 'gymnasium/envs/registration.py')
if patched:
    importlib.reload(gymnasium)
    importlib.reload(_reg)
print("✅ Gymnasium patched" if patched else "✅ Gymnasium OK")

# ============================================================================
# 2. IMPORTS (after path + patch setup)
# ============================================================================
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pettingzoo import ParallelEnv
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent

# ============================================================================
# 3. SOTA ENV WRAPPER (Red team = heuristic bots handled inside env)
# ============================================================================
# NOTE: This class MUST be in an importable file for Ray workers.
# We write it to pyquaticus/envs/sota_wrapper.py at runtime.

_WRAPPER_CODE = '''
import functools
import numpy as np
from pettingzoo import ParallelEnv
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent

class SotaEnvWrapper(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "pyquaticus_sota"}

    def __init__(self, config_dict, reward_config, team_size=3,
                 render_mode=None, opponent_mode="hard"):
        super().__init__()
        self.base_env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict, render_mode=render_mode,
            reward_config=reward_config, team_size=team_size)
        self.blue_agents = ["agent_0", "agent_1", "agent_2"]
        self.red_agents = ["agent_3", "agent_4", "agent_5"]
        self.possible_agents = list(self.blue_agents)
        self.agents = list(self.blue_agents)
        self.red_policies = {
            "agent_3": BaseAttacker("agent_3", self.base_env, mode=opponent_mode),
            "agent_4": BaseDefender("agent_4", self.base_env, mode=opponent_mode),
            "agent_5": Heuristic_CTF_Agent("agent_5", self.base_env, mode=opponent_mode),
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
        return ({k: v for k, v in obs.items() if k in self.blue_agents},
                {k: v for k, v in infos.items() if k in self.blue_agents})

    def step(self, blue_actions):
        red_actions = {}
        for aid in self.red_agents:
            if aid in self._all_obs:
                try:
                    red_actions[aid] = self.red_policies[aid].compute_action(
                        self._all_obs[aid], self._all_infos.get(aid, {}))
                except Exception:
                    red_actions[aid] = self.base_env.action_space(aid).sample()
        all_actions = {**blue_actions, **red_actions}
        obs, rews, terms, truncs, infos = self.base_env.step(all_actions)
        self._all_obs = obs
        self._all_infos = infos
        self.agents = [a for a in self.blue_agents if a in obs]
        B = self.blue_agents
        return ({k: v for k, v in obs.items() if k in B},
                {k: v for k, v in rews.items() if k in B},
                {k: v for k, v in terms.items() if k in B},
                {k: v for k, v in truncs.items() if k in B},
                {k: v for k, v in infos.items() if k in B})

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
'''

# Write wrapper to importable location
_wrapper_path = os.path.join(_repo_root, 'pyquaticus', 'envs', 'sota_wrapper.py')
if not os.path.exists(_wrapper_path):
    with open(_wrapper_path, 'w') as f:
        f.write(_WRAPPER_CODE)
    print("✅ Created pyquaticus/envs/sota_wrapper.py")
else:
    print("✅ sota_wrapper.py exists")

from pyquaticus.envs.sota_wrapper import SotaEnvWrapper

# ============================================================================
# 4. MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCTF SOTA 3v3 Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path or "latest" to auto-find')
    parser.add_argument('--max-iter', type=int, default=8000)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--backup-dir', type=str,
                        default='/kaggle/working/mctf_checkpoints')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)

    # --- Environment Config ---
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
        'agent_3': None, 'agent_4': None, 'agent_5': None
    }

    render_mode = 'human' if args.render else None

    def env_creator(config):
        return SotaEnvWrapper(
            config_dict=config_dict,
            reward_config=reward_config,
            team_size=3,
            render_mode=render_mode,
            opponent_mode='hard'
        )

    env = ParallelPettingZooWrapper(env_creator({}))
    register_env('pyquaticus_sota',
                 lambda config: ParallelPettingZooWrapper(env_creator(config)))

    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    env.close()

    # --- PPO Config ---
    policies = {'learned_policy': (None, obs_space, act_space, {})}

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
            policy_mapping_fn=lambda aid, *a, **kw: "learned_policy",
            policies_to_train=["learned_policy"]
        )
    )

    # --- Build / Resume ---
    os.makedirs(args.backup_dir, exist_ok=True)
    start_iter = 0

    if args.resume:
        # Find latest checkpoint
        if args.resume == 'latest':
            checkpoints = sorted([
                d for d in os.listdir(args.backup_dir)
                if d.startswith('iter_') and
                os.path.isdir(os.path.join(args.backup_dir, d))
            ])
            if checkpoints:
                args.resume = os.path.join(args.backup_dir, checkpoints[-1])
                start_iter = int(checkpoints[-1].split('_')[1]) + 1
                print(f"📂 Auto-found latest: {args.resume} (iter {start_iter})")
            else:
                print("⚠️  No checkpoints found, starting fresh")
                args.resume = None

        if args.resume:
            algo = ppo_config.build_algo()
            algo.restore(args.resume)
            print(f"📂 Resumed from {args.resume}")
        else:
            algo = ppo_config.build_algo()
    else:
        algo = ppo_config.build_algo()

    # --- Training ---
    print("\n" + "="*70)
    print(f"🚀 MCTF SOTA TRAINING")
    print(f"🔴 Red: Attacker(hard) + Defender(hard) + Combined(hard)")
    print(f"🔵 Blue: PPO shared policy (GPU)")
    print(f"📊 Iterations: {start_iter} → {args.max_iter}")
    print(f"💾 Saving every {args.save_interval} iters to {args.backup_dir}")
    print("="*70 + "\n")

    for i in range(start_iter, args.max_iter):
        t0 = time.time()
        result = algo.train()
        dt = time.time() - t0

        er = result.get('env_runners', {})
        rm = er.get('episode_reward_mean', 0)
        rmin = er.get('episode_reward_min', 0)
        rmax = er.get('episode_reward_max', 0)
        el = er.get('episode_len_mean', 0)

        print(f"Iter {i:>5}/{args.max_iter} | "
              f"R={rm:>8.2f} ({rmin:.1f}~{rmax:.1f}) | "
              f"Len={el:>5.0f} | {dt:.1f}s")

        if i % args.save_interval == 0:
            chkpt = f'./ray_test/iter_{i}/'
            algo.save(chkpt)
            bk = os.path.join(args.backup_dir, f'iter_{i}')
            if os.path.exists(bk):
                shutil.rmtree(bk)
            shutil.copytree(chkpt, bk)
            print(f"   💾 Saved!")

    algo.save('./ray_test/iter_final/')
    print(f"\n🎉 Done! Model saved to {args.backup_dir}")
