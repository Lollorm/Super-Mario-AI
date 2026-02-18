# Super Mario AI
>This repository explores multiple AI approaches to training AI agents to play Super Mario Bros and Super Mario World.

---

![mario1-1](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/mario1-1.gif)

An AI agent trained with the NeuroEvolution of Augmenting Topologies (NEAT) algorithm successfully beating Level 1-1  
(Even going as far as learning a frame/pixel perfect glitch called 'wall jump')

<img src="https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/fitness_progress_0_to_350.png" width="800"/>

Best fitness over generations of the previous agent (The best individual of generation 352 managed to beat the level).

- [NEAT Implementation](#neat-implementation)
  - [About NEAT](#about-neat)
  - [Environment](#environment)
  - [Requirements](#requirements)
  - [Defining an Input](#defining-an-input)
  - [Defining a Fitness Function](#defining-a-fitness-function)
- [PPO Implementation](#ppo-implementation)
  - [About PPO](#about-ppo)
  - [Requirements](#requirements-1)
  - [Environment](#environment-1)
  - [Network Architecture](#network-architecture)
  - [Training on Random Levels](#training-on-random-levels)
  - [Donut Plains 3](#donut-plains-3)
- [DDQN Implementation (WIP)](#ddqn-implementation)

---

## NEAT implementation

### About NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm used to evolve artificial neural networks. It gradually optimizes both network structure and weights through evolutionary processes. A full description of the algorithm is available in the original [paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).  
In this project, NEAT is implemented using the NEAT-Python library. The official documentation can be found [here](https://neat-python.readthedocs.io/en/latest/).

![net](https://github.com/Lollorm/Super-Mario-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/mario_net.png)

Here's the topology of the best evolved network from generation 352.

---

### Environment

The Gym Super Mario Bros environment was used to provide the game interface and training environment for the agent. Documentation can be found [here](https://pypi.org/project/gym-super-mario-bros/).

```
pip install gym-super-mario-bros
```

#### To play manually:

```
gym_super_mario_bros -e 'SuperMarioBrosRandomStages-v0' -m 'human' --stages '1-4'
```

#### Requirements

```
pip install gym==0.25.1 gym-super-mario-bros==7.4.0 nes-py==8.2.1 neat-python==1.1.0 numpy==1.26.4 opencv-python==4.12.0.88
```

---

### Defining an input

To select the optimal action, the neural network is fed the current environment observation at each timestep. In this implementation, the agent captures a screenshot of the current frame, converts it to grayscale, normalizes pixel values to the range [0,1], and reduces the image to a fixed-size representation of 1024 inputs. The network is configured as recurrent, so temporal information is handled through the network’s internal state rather than explicit frame stacking. As a result, no frame stacking is used. In this particular case, experiments with frame stacking using a similar number of inputs proved ineffective.

![screenshot](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/assets/images/statescreenshot_resized.png)
![screenshot](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/assets/images/statescreenshot.png)

The input image and its magnified version side by side.

---

### Defining a Fitness Function

When working with genetic algorithms, it all really comes down to defining a fitness function that accurately evaluates how well individuals perform a given task.  
In particular, if we want to train an AI agent to, let's say, clear level 1-1 of Super Mario Bros, we should analyze what it really means to be a good player in Super Mario.

![mario_jump](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/assets/images/mario_jump.gif)

Here's an example of what a poorly defined fitness function can lead to (in this case jumping was rewarded too much).

In this implementation, the fitness function evaluates Mario’s speed and position along the x-axis. Penalties are applied if Mario gets stuck or dies. In addition, a small reward is given for jumping, to encourage the evolution of individuals who jump obstacles and advance further.

---

### Fitness Function
Let $t = 1, \dots, T$ denote timesteps until termination. The total fitness $F$ of an individual is defined as:
```math
F = \max \Bigg(
0.1,\;
\sum_{t=1}^{T} \Big( r_t + 0.01 + 0.1 \,(x_t - x_{t-1}) + 0.1 \,\mathbf{}_{y_t > y_{t-1}} \Big)
+ \max_t(x_t)
+ R_{\text{term}}
\Bigg)
```

where the terminal reward $R_{\text{term}}$ is:
```math
R_{\text{term}} =
\begin{cases}
+10000 & \text{if the flag is reached} \\
-150 & \text{if the agent dies before reaching the flag} \\
-100 & \text{if the agent gets stuck}
\end{cases}
```

- $r_t$ is the environment reward from *gym\_super\_mario\_bros*  
- $x_t, y_t$ are Mario's horizontal and vertical positions at timestep $t$  
- $\mathbf{}{y_t > y_{t-1}}$ is an indicator function of whether Mario is jumping or not  
- "stuck" is defined as no forward progress for more than 250 consecutive timesteps
- such a huge reward for reaching the flagpole is useful to understand in which generation the agent managed to beat the level

---

### Try it Yourself

The code to train the agent can be found in the [`NEAT`](NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/) folder in this repository, it can be easily modified to change the behaviour of the agent, for example you can penalize slow agents to encourage the evolution of faster specimens that can reach speeds on par with glitchless world records.

---

## PPO implementation

### About PPO

![Yoshi Island 2](https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/Assets/YoshiIsland2.gif)

An agent trained only on Yoshi Island 1 beating Yoshi Island 2 (look at what he does with the green koopa).

Proximal policy optimization (PPO) is a reinforcement learning algorithm developed by researchers at OpenAI, specifically, it is a policy gradient method.
It has proven effective in challenging environments, for example an agent trained with this algorithm (OpenAI Five) was the first AI to [beat the world champions in an esports game](https://openai.com/index/openai-five-defeats-dota-2-world-champions/), and therefore it is perfect for Mario.
Unlike off-policy algorithms such as DDQN, PPO does not use a replay buffer. As an on-policy method, PPO continuously updates its policy using data generated by the current policy only.

This time, I implemented it using stable-baselines3 [library](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and following ClarityCoders's [tutorial](https://youtu.be/PxoG0A2QoFs?si=lWY_leDz14ngZk5S) on PPO and adapting it to Super Mario World.

If you want to learn more about this algorithm, the original paper that describes it can be found [here](https://arxiv.org/pdf/1707.06347v2).

**Note**  
The trained model zip files are approximately 260 MB and cannot be uploaded to GitHub due to size limits.


---

### Requirements

You can import the requirements from [this](https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/requirements.txt) file.  
This time I am using python 3.8.10 and pip 22.3.1

---

### Environment

This time I used OpenAI's [Gym Retro](https://openai.com/index/gym-retro/) environment because it supports custom levels and allows direct access to RAM values, making it possible to inspect memory locations during gameplay.  
This time I decided to train the agent on [Super Mario World](https://en.wikipedia.org/wiki/Super_Mario_World) as it offers a more complex action space, in particular I was curious about the possibility of an agent learning trickjumps or glitches (Maybe an Agent could learn [Arbitrary code execution](https://tasvideos.org/ArbitraryCodeExecutionHowTo) given the proper environment).  

To do this I had to manually [integrate](https://retro.readthedocs.io/en/latest/integration.html) the game in Gym Retro by:  
1. Finding the memory addresses of relevant variables (position, lives, score, etc.)
2. Implementing a custom reward function
3. Adding the necessary JSON configuration files

![Yoshi Island 1](https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/Assets/YoshiIsland1.gif)

The same agent from before beating Yoshi Island 1.  


To implement the agent you will have to replace the files (with the same name) in your game directory with [these](https://github.com/Lollorm/Super-Mario-AI/tree/main/PPO%20Proximal%20Policy%20Optimization/JSON) JSON files.  


For example in [scenario.json](https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/JSON/scenario.json) you can change or tweak these parameters, and add others to influence the reward function, in particular you can choose to feed the neural network only part of the screen by modyfing "crop" (In my case the network received the full image as input). I also decided to end each episode (done) after the player died once, even though Mario starts with 5 lives.
```
{
  "crop": [
    0,
    0,
    0,
    0
  ],
  "done": {
    "variables": {
      "dead": {
        "op": "equal",
        "reference": 0
      }
    }
  },
  "reward": {
    "variables": {
      "mario_x": {
        "reward": 1.0
      },
      "score": {
        "reward": 0.01
      },
      "checkpoint": {
        "op": "delta",
        "measurement": "nonzero",
        "reward": 100.0
      },
      "level_end": {
        "reward": 2000.0
    
      }
    }
  }
}
```
<img src="https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/Assets/GameIntegrationGUI.png" width="500"/>

Me searching for the memory address for holding a shell.

---

### Network Architecture

This PPO implementation uses a Convolutional Neural Network (CNN) policy to process visual observations directly from the game screen.  
You can visualize the architecture by printing [model.policy](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```
ActorCriticCnnPolicy(
  (features_extractor): NatureCNN(
    (cnn): Sequential(
      (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
      (1): ReLU()
      (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
      (3): ReLU()
      (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
      (5): ReLU()
      (6): Flatten(start_dim=1, end_dim=-1)
    )
    (linear): Sequential(
      (0): Linear(in_features=43008, out_features=512, bias=True)
      (1): ReLU()
    )
  )
  (mlp_extractor): MlpExtractor(
    (shared_net): Sequential()
    (policy_net): Sequential()
    (value_net): Sequential()
  )
  (action_net): Linear(in_features=512, out_features=12, bias=True)
  (value_net): Linear(in_features=512, out_features=1, bias=True)
)
```
---

### Training On Random Levels

![RandomLevelTraining](https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/Assets/YoshiIsland2RandomLevelTraining.gif)

An agent trained on random levels beating Yoshi Island 2.

To encourage generalization, the agent can be trained across a list of similar levels rather than a single stage. During training, a level is randomly selected at the start of each episode. This setup prevents overfitting to a specific environment and forces the agent to learn strategies that generalize across all levels, resulting in more robust (successful on a variety of problems without hyperparameter tuning) and versatile behavior.  

[Here](https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/Training/TrainAgentOnRandomLevels.py) is my implementation of this training setup.

---

### Donut Plains 3

<img src="https://github.com/Lollorm/Super-Mario-AI/blob/main/PPO%20Proximal%20Policy%20Optimization/Assets/DonutPlains4Best.gif" width="500"/>

An agent trained on Donut Plains 3 (In gym-retro it is called Donut Plains 4).

Donut Plains 3 is a challenging level for an AI as it includes moving platforms. After a bit of trial and error the agent manages to advance in the level surprisingly well, sadly in the middle it tries to use the on/off switch but it fails to wait for the moving platform. Nonetheless it's still impressive how an agent trained with PPO manages to adapt to moving platforms in such a short time. it's possible that by giving it enough time and by shaping the reward function to also encourage waiting for the platform, an AI agent could beat this level. 

---

## DDQN implementation

### $${\color{red}WIP}$$

<img src="https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/DDQN/assets/images/cartpole_qlearning-episode-0.gif" width="500"/>

An agent I made in the past balancing an inverted pendulum (using a Q-table).

---

### About DDQN

[paper](https://arxiv.org/pdf/1509.06461)

---

### Acknowledgments




