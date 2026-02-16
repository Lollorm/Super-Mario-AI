import retro
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import gym
import os
from datetime import datetime
from TrainAgentOnRandomLevels import TimeLimitWrapper

# Custom wrapper that captures ALL frames while still doing frame skipping for the model
class MaxAndSkipEnvWithRecording(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip
        self.recorded_frames = []  # Store all frames for recording
    
    def reset(self):
        self.recorded_frames = []
        return self.env.reset()
    
    def step(self, action):
        total_reward = 0.0
        done = None
        self.recorded_frames = []
        
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            
            frame = self.env.render(mode='rgb_array')
            self.recorded_frames.append(frame)
            
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        
        max_frame = self._obs_buffer.max(axis=0)
        
        return max_frame, total_reward, done, info


# Create videos directory if it doesn't exist
os.makedirs("videos", exist_ok=True)

# Generate unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"videos/mario_ppo_hq_60fps_{timestamp}.mp4"

# Load trained model
model = PPO.load("tmp/best_model.zip")

# Create environment
env = retro.make(
    game="SuperMarioWorld-Snes",
    state="DonutPlains4"
)
env = TimeLimitWrapper(env)
env = MaxAndSkipEnvWithRecording(env, skip=4)

obs = env.reset()

frame = env.render(mode="rgb_array")
h, w, _ = frame.shape

scale = 5
out_w, out_h = w * scale, h * scale

fourcc = cv2.VideoWriter_fourcc(*"avc1")
video = cv2.VideoWriter(
    video_filename,
    fourcc,
    60,  
    (out_w, out_h)
)

done = False
frame_count = 0

print("Recording gameplay...")

while not done:
    
    action, _ = model.predict(obs)  # deterministic=True if you want to see the same gameplay every time
    obs, reward, done, info = env.step(action)
    
    
    for frame in env.recorded_frames:
        
        frame_scaled = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        
        frame_bgr = cv2.cvtColor(frame_scaled, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        video.write(frame_bgr)
        
        frame_count += 1
    
    if frame_count % 1000 == 0:
        print(f"Recorded {frame_count} frames...")

print(f"Total frames recorded: {frame_count}")
print(f"Video duration: {frame_count / 60:.2f} seconds")

video.release()
env.close()

print(f"Video saved successfully")
print(f"Location: {video_filename}")
