import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import neat
import os
import numpy as np
import pickle
import multiprocessing

max_steps = 5000
max_generations = 2500
generation = -1


def create_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, [["right","B"], ["right", "A","B"]])
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (32, 32))
    return env


def eval_genome(genome, config):
    env = create_env()
    state = env.reset()
    fitness = 0
    previous_xpos = 0
    previous_ypos = 0
    stuck_counter = 0
    max_x = 0
    max_y = 0
    done = False

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    while not done:
        state_flat = np.array(state).flatten() / 255.0
        output = net.activate(state_flat)
        action = int(np.argmax(output))
        next_state, reward, done, info = env.step(action)

        xpos = info["x_pos"]
        ypos = info["y_pos"]
        delta_x = xpos - previous_xpos
        delta_y = ypos - previous_ypos
        max_y = max(ypos, max_y)
        fitness += reward
        "fitness += reward + delta_x"
        fitness += 0.01
        fitness += delta_x * 0.1
        if ypos > previous_ypos:
            fitness += 0.1
        max_x = max(xpos, max_x)

        """if delta_y > 0:
            fitness += 0.1"""
        

        if delta_x <= 0:
            stuck_counter += 1
        else:
            stuck_counter = 0

        if stuck_counter > 250:
            fitness -= 100
            break

        if done and not info.get('flag_get', False):
            fitness -= 150
            break

        if info.get('flag_get', False):
            fitness += 10000
            break

        previous_xpos = xpos
        previous_ypos = ypos
        state = next_state

        if done:
            break

    env.close()
    fitness += max_x
    return max(fitness, 0.1)


def run(config_file):
    
    global generation

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Parallel evaluation using all CPU cores
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    """winner = p.run(pe.evaluate, max_generations)"""
    winner = p.run(pe.evaluate)

    # Save winner genome
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
        print("Saved winner genome")

    print(f'\nBest genome:\n{winner!s}')
    print(f'Best fitness: {winner.fitness}')


def test_genome(genome, config, render=True):
    """Test a single genome in the environment."""
    env = create_env()
    state = env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_reward = 0
    max_x = 0

    for _ in range(max_steps):
        if render:
            env.render()

        state_flat = np.array(state).flatten() / 255.0
        output = net.activate(state_flat)
        action = int(np.argmax(output))
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        max_x = max(max_x, info.get("x_pos", 0))
        state = next_state

        if done:
            break

    env.close()
    print(f"Test - Total Reward: {total_reward}, Max X Position: {max_x}")
    return total_reward, max_x


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please ensure 'config.ini' exists in the same directory as this script")
    else:
        run(config_path)
