#For quick testing of your environment
import retro
import gym

env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains4')
obs = env.reset()

print(obs.shape)

done = False

while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
    
env.close()