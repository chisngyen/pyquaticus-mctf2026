# ============================================================================
# MCTF 2026 - SOTA Training (Paste & Run in Kaggle)
# ============================================================================
import sys, os, warnings, shutil, time, logging
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === STEP 1: CLONE REPO (must happen before any pyquaticus imports) ===
REPO_URL = "https://github.com/chisngyen/pyquaticus-mctf2026.git"
REPO_DIR = "/kaggle/working/pyquaticus"
CHECKPOINT_DIR = "/kaggle/working/mctf_checkpoints"
MAX_ITER = 2000
SAVE_INTERVAL = 100

if not os.path.exists(REPO_DIR):
    print(f"📥 Cloning {REPO_URL}...")
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    os.system("pip install -q pettingzoo pygame shapely scipy lz4")
else:
    print(f"✅ Repo exists")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# === STEP 2: PATCH GYMNASIUM *BEFORE* importing Ray RLlib ===
import importlib
import gymnasium
import gymnasium.envs.registration as _reg

_gym_init = os.path.join(os.path.dirname(gymnasium.__file__), '__init__.py')
_reg_file = os.path.join(os.path.dirname(_reg.__file__), 'registration.py')

_patch_code = '\nimport enum as _patch_enum\nclass VectorizeMode(_patch_enum.Enum):\n    SYNC="sync"\n    ASYNC="async"\n    AUTORESET="autoreset"\n'

_patched = False
for fpath, label in [(_gym_init, 'gymnasium'), (_reg_file, 'registration')]:
    with open(fpath, 'r') as f:
        if 'VectorizeMode' not in f.read():
            with open(fpath, 'a') as f2:
                f2.write(_patch_code)
            _patched = True
            print(f"🔧 Patched {label}")

if _patched:
    importlib.reload(gymnasium)
    importlib.reload(_reg)
print("✅ Gymnasium OK")

# === STEP 3: NOW import Ray (after patch) ===
import numpy as np
import ray

# Initialize Ray with PYTHONPATH so workers can find pyquaticus
ray.init(runtime_env={"env_vars": {"PYTHONPATH": REPO_DIR}}, ignore_reinit_error=True)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pettingzoo import ParallelEnv

# === STEP 4: Import pyquaticus ===
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent

# === STEP 5: CREATE WRAPPER FILE (Ray workers need importable file) ===
WRAPPER_FILE = os.path.join(REPO_DIR, "pyquaticus/envs/sota_wrapper_kaggle.py")
with open(WRAPPER_FILE, "w") as f:
    f.write('''
import functools, numpy as np
from pettingzoo import ParallelEnv
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent

class SotaEnvWrapper(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "pyquaticus_sota"}
    def __init__(self, config_dict, reward_config, team_size=3, render_mode=None, opponent_mode="hard"):
        super().__init__()
        self.base_env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict, render_mode=render_mode, reward_config=reward_config, team_size=team_size)
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
    def observation_space(self, agent): return self.base_env.observation_space(agent)
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self.base_env.action_space(agent)
    def reset(self, seed=None, options=None):
        obs, infos = self.base_env.reset(seed=seed, options=options)
        self._all_obs, self._all_infos = obs, infos
        self.agents = [a for a in self.blue_agents if a in obs]
        return {k: v for k, v in obs.items() if k in self.blue_agents}, {k: v for k, v in infos.items() if k in self.blue_agents}
    def step(self, blue_actions):
        red_actions = {}
        for aid in self.red_agents:
            if aid in self._all_obs:
                try: red_actions[aid] = self.red_policies[aid].compute_action(self._all_obs[aid], self._all_infos.get(aid, {}))
                except: red_actions[aid] = self.base_env.action_space(aid).sample()
        obs, rews, terms, truncs, infos = self.base_env.step({**blue_actions, **red_actions})
        self._all_obs, self._all_infos = obs, infos
        self.agents = [a for a in self.blue_agents if a in obs]
        B = self.blue_agents
        return ({k:v for k,v in obs.items() if k in B}, {k:v for k,v in rews.items() if k in B},
                {k:v for k,v in terms.items() if k in B}, {k:v for k,v in truncs.items() if k in B},
                {k:v for k,v in infos.items() if k in B})
    def render(self): return self.base_env.render()
    def close(self): self.base_env.close()
''')
print("✅ Wrapper created")
from pyquaticus.envs.sota_wrapper_kaggle import SotaEnvWrapper

# === STEP 6: TRAIN ===
logging.basicConfig(level=logging.ERROR)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

config_dict = config_dict_std.copy()
config_dict.update({'sim_speedup_factor': 4, 'max_score': 3, 'max_time': 240, 'tagging_cooldown': 60, 'tag_on_oob': True})
reward_config = {'agent_0': rew.caps_and_grabs, 'agent_1': rew.caps_and_grabs, 'agent_2': rew.caps_and_grabs, 'agent_3': None, 'agent_4': None, 'agent_5': None}

def env_creator(config):
    return SotaEnvWrapper(config_dict=config_dict, reward_config=reward_config, team_size=3, opponent_mode='hard')

register_env('pyquaticus_sota', lambda config: ParallelPettingZooWrapper(env_creator(config)))
dummy_env = ParallelPettingZooWrapper(env_creator({}))
obs_space, act_space = dummy_env.observation_space['agent_0'], dummy_env.action_space['agent_0']
dummy_env.close()

ppo_config = (
    PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(env='pyquaticus_sota')
    .framework("torch").resources(num_gpus=1)
    .env_runners(num_env_runners=2, num_cpus_per_env_runner=1, sample_timeout_s=600.0)
    .multi_agent(policies={'learned_policy': (None, obs_space, act_space, {})},
                 policy_mapping_fn=lambda aid, *a, **kw: "learned_policy",
                 policies_to_train=["learned_policy"])
)

print("\n" + "="*60)
print(f"🚀 TRAINING: 0 -> {MAX_ITER} | Save every {SAVE_INTERVAL}")
print("🔴 Red: Attacker + Defender + Combined (hard)")
print("🔵 Blue: PPO shared policy (GPU)")
print("="*60 + "\n")

algo = ppo_config.build_algo()

for i in range(MAX_ITER):
    t0 = time.time()
    result = algo.train()
    dt = time.time() - t0
    er = result.get('env_runners', {})
    print(f"Iter {i:4d} | R={er.get('episode_reward_mean',0):6.2f} | Len={er.get('episode_len_mean',0):4.0f} | {dt:4.1f}s")
    if i % SAVE_INTERVAL == 0:
        chkpt = algo.save()
        print(f"   💾 Saved: {chkpt}")

print("🎉 DONE!")
