# ============================================================================
# MCTF 2026 - SOTA Training Cell (Paste into Kaggle)
# ============================================================================
# SELF-CONTAINED: clones, patches, and trains.
# Red team handled INSIDE env wrapper (no info_batch issues).
# SotaEnvWrapper lives in pyquaticus/envs/sota_wrapper.py (importable by workers).
# ============================================================================

import sys, os, warnings, importlib
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === STEP 1: Patch gymnasium ON DISK for Ray workers ===
import gymnasium
_gym_init = os.path.join(os.path.dirname(gymnasium.__file__), '__init__.py')
with open(_gym_init, 'r') as f:
    _content = f.read()
if 'VectorizeMode' not in _content:
    with open(_gym_init, 'a') as f:
        f.write('''
import enum as _patch_enum
class VectorizeMode(_patch_enum.Enum):
    SYNC = "sync"
    ASYNC = "async"
    AUTORESET = "autoreset"
''')
    print(f"🔧 Patched gymnasium/__init__.py")
    importlib.reload(gymnasium)
else:
    print("✅ gymnasium already patched")

import gymnasium.envs.registration as _reg
_reg_file = os.path.join(os.path.dirname(_reg.__file__), 'registration.py')
with open(_reg_file, 'r') as f:
    _content = f.read()
if 'VectorizeMode' not in _content:
    with open(_reg_file, 'a') as f:
        f.write('''
import enum as _patch_enum
class VectorizeMode(_patch_enum.Enum):
    SYNC = "sync"
    ASYNC = "async"
    AUTORESET = "autoreset"
''')
    print(f"🔧 Patched registration.py")
    importlib.reload(_reg)
else:
    print("✅ registration.py already patched")

# === STEP 2: Clone repo ===
REPO_DIR = '/kaggle/working/pyquaticus'
if not os.path.exists(REPO_DIR):
    print("\n📥 Cloning repository...")
    os.system('git clone https://github.com/technoob05/pyquaticus.git ' + REPO_DIR)
    os.chdir(REPO_DIR)
    os.system('git checkout mctf2026')
else:
    print(f"✅ Repo at {REPO_DIR}")
    os.chdir(REPO_DIR)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.system('pip install -q pettingzoo pygame shapely scipy lz4 2>/dev/null')

# === STEP 2.5: Create SotaEnvWrapper file (so Ray workers can import it) ===
WRAPPER_PATH = os.path.join(REPO_DIR, 'pyquaticus', 'envs', 'sota_wrapper.py')
if not os.path.exists(WRAPPER_PATH):
    print("📝 Creating sota_wrapper.py...")
    with open(WRAPPER_PATH, 'w') as f:
        f.write('''
import functools
import numpy as np
from pettingzoo import ParallelEnv
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent

class SotaEnvWrapper(ParallelEnv):
    """Red team controlled by heuristic bots internally. RLlib only sees Blue."""
    metadata = {"render_modes": ["human"], "name": "pyquaticus_sota"}

    def __init__(self, config_dict, reward_config, team_size=3,
                 render_mode=None, opponent_mode='hard'):
        super().__init__()
        self.base_env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict, render_mode=render_mode,
            reward_config=reward_config, team_size=team_size)
        self.opponent_mode = opponent_mode
        self.blue_agents = ['agent_0', 'agent_1', 'agent_2']
        self.red_agents = ['agent_3', 'agent_4', 'agent_5']
        self.possible_agents = list(self.blue_agents)
        self.agents = list(self.blue_agents)
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
        return {k: v for k, v in obs.items() if k in self.blue_agents}, \\
               {k: v for k, v in infos.items() if k in self.blue_agents}

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
        obs, rewards, terms, truncs, infos = self.base_env.step(all_actions)
        self._all_obs = obs
        self._all_infos = infos
        self.agents = [a for a in self.blue_agents if a in obs]
        return ({k: v for k, v in obs.items() if k in self.blue_agents},
                {k: v for k, v in rewards.items() if k in self.blue_agents},
                {k: v for k, v in terms.items() if k in self.blue_agents},
                {k: v for k, v in truncs.items() if k in self.blue_agents},
                {k: v for k, v in infos.items() if k in self.blue_agents})

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
''')
    print("✅ sota_wrapper.py created!")
else:
    print("✅ sota_wrapper.py exists")

# === STEP 3: Imports ===
import numpy as np
import ray
import time
import shutil
import logging
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus.envs.sota_wrapper import SotaEnvWrapper
from pyquaticus.config import config_dict_std
import pyquaticus.utils.rewards as rew

logging.basicConfig(level=logging.ERROR)

# === CONFIGURATION ===
MAX_ITER = 8000
SAVE_INTERVAL = 100
CHECKPOINT_DIR = '/kaggle/working/mctf_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

# === SETUP ===
print("\n" + "="*70)
print("🚀 MCTF SOTA TRAINING")
print("="*70)

def env_creator(config):
    return SotaEnvWrapper(
        config_dict=config_dict,
        reward_config=reward_config,
        team_size=3,
        render_mode=None,
        opponent_mode='hard'
    )

dummy_env = ParallelPettingZooWrapper(env_creator({}))
register_env('pyquaticus_sota', lambda config: ParallelPettingZooWrapper(env_creator(config)))

obs_space = dummy_env.observation_space['agent_0']
act_space = dummy_env.action_space['agent_0']
dummy_env.close()

policies = {
    'learned_policy': (None, obs_space, act_space, {}),
}

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "learned_policy"

print("🔴 Red Team: Attacker(hard) + Defender(hard) + Combined(hard) [in-env]")
print("🔵 Blue Team: PPO (shared policy, GPU)")

# === BUILD PPO ===
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

algo = ppo_config.build_algo()
print(f"\n🚀 Training 0 → {MAX_ITER} | Saving every {SAVE_INTERVAL} iters")
print("="*70 + "\n")

# === TRAINING LOOP ===
for i in range(MAX_ITER):
    start_time = time.time()
    result = algo.train()
    elapsed = time.time() - start_time

    er = result.get('env_runners', {})
    reward_mean = er.get('episode_reward_mean', 0)
    reward_min  = er.get('episode_reward_min', 0)
    reward_max  = er.get('episode_reward_max', 0)
    ep_len      = er.get('episode_len_mean', 0)

    print(f"Iter {i:>5}/{MAX_ITER} | "
          f"Reward: {reward_mean:>8.2f} (min={reward_min:.2f}, max={reward_max:.2f}) | "
          f"EpLen: {ep_len:>6.0f} | Time: {elapsed:>5.1f}s")

    if i % SAVE_INTERVAL == 0:
        chkpt = f'./ray_test/iter_{i}/'
        algo.save(chkpt)
        backup = os.path.join(CHECKPOINT_DIR, f'iter_{i}')
        if os.path.exists(backup):
            shutil.rmtree(backup)
        shutil.copytree(chkpt, backup)
        print(f"   💾 Checkpoint saved!")

final_path = './ray_test/iter_final/'
algo.save(final_path)
shutil.copytree(final_path, os.path.join(CHECKPOINT_DIR, 'iter_final'), dirs_exist_ok=True)
print(f"\n🎉 Done! Model: {CHECKPOINT_DIR}/iter_final/")
