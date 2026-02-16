#run this line to import the game roms in retro
#python -m retro.import "C:\Users\yourfolderdirectory\gamefolder"

#run this line to visualize real time training data in tensorboard
#tensorboard --logdir "board"

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import os

import retro
import random

class TimeLimitWrapper(gym.Wrapper):
  def __init__(self, env, max_steps=10000):
    super(TimeLimitWrapper, self).__init__(env)
    self.max_steps = max_steps
    self.current_step = 0
  
  def reset(self):
    self.current_step = 0
    return self.env.reset()

  def step(self, action):
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    if self.current_step >= self.max_steps:
      done = True
      info['time_limit_reached'] = True
    info['Current_Step'] = self.current_step
    return obs, reward, done, info

class RandomLevelWrapper(gym.Env):
    def __init__(self, game, states):
        super().__init__()
        self.game = game
        self.states = states
        self.env = None

        dummy = retro.make(game=game, state=states[0])
        self.observation_space = dummy.observation_space
        self.action_space = dummy.action_space
        dummy.close()

    def reset(self):
        if self.env is not None:
            self.env.close()

        state = random.choice(self.states)
        self.env = retro.make(game=self.game, state=state)
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        if self.env is not None:
            self.env.close()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)



def make_env(env_id, rank, seed=0):

    #choose the levels to train on
    LEVELS = [
    "YoshiIsland1",
    "YoshiIsland2",
    "DonutPlains1",
    "DonutPlains5",
    ]
    
    def _init():
        states = LEVELS
        #if you want to train on your list of levels use this line
        env = RandomLevelWrapper(env_id, states)
        #if you want to train on a single level use this line
        #env = retro.make(game=env_id, state='DonutPlains4')
        env = TimeLimitWrapper(env, max_steps=2000)
        env = MaxAndSkipEnv(env, 4)

        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_id = 'SuperMarioWorld-Snes'
    num_cpu = 4
    
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]),"tmp/TestMonitor")

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003)
    #if you want to resume training use this line
    #model = PPO.load("tmp/best_model.zip", env=env)
    
    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=5000000, callback=callback, tb_log_name="PPO-00003")
    model.save(env_id)
    print("------------- Done Learning -------------")
    env = retro.make(game=env_id)
    env = TimeLimitWrapper(env)
    
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
