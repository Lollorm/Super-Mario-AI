# Super Mario Bros AI

# CURRENTLY WIP

![mario1-1](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/mario1-1.gif)

An AI agent trained with the NeuroEvolution of Augmenting Topologies (NEAT) algorithm successfully beating Level 1-1  
(Even going as far as learning a frame/pixel perfect glitch called 'wall jump')

![Best fitness over generations](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/fitness_progress_0_to_350.png)

Best fitness over generations



## NEAT implementation

### About NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm used to evolve artificial neural networks. It gradually optimizes both network structure and weights through evolutionary processes. A full description of the algorithm is available in the original [paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).  
In this project, NEAT is implemented using the NEAT-Python library. The official documentation can be found [here](https://neat-python.readthedocs.io/en/latest/).

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

### Defining an input

To select the optimal action, the neural network is fed the current environment observation at each timestep. In this implementation, the agent captures a screenshot of the current frame, converts it to grayscale, normalizes pixel values to the range [0,1], and reduces the image to a fixed-size representation of 1024 inputs. The network is configured as recurrent, so temporal information is handled through the network’s internal state rather than explicit frame stacking. As a result, no frame stacking is used. In this particular case, experiments with frame stacking using a similar number of inputs proved ineffective.

### Defining a Fitness Function

When working with genetic algorithms, it all really comes down to defining a fitness function that accurately evaluates how well individuals perform a given task.  
In particular, if we want to train an AI agent to, let's say, clear level 1-1 of Super Mario Bros, we should analyze what it really means to be a good player in Super Mario.

![mario_jump](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/assets/images/mario_jump.gif)

Here's an example of what a poorly defined fitness function can lead to.

In this implementation, the fitness function evaluates Mario’s speed and position along the x-axis. Penalties are applied if Mario gets stuck or dies. In addition, a small reward is given for jumping, to encourage the evolution of individuals who jump obstacles and advance further.

F = max(
0.1,
sum_{t=1}^{T} ( r_t + 0.01 + 0.1 * (x_t - x_{t-1}) + 0.1 * 1_{y_t > y_{t-1}} )
+ max_t(x_t)
+ R_term
)

work in progress

## DDQN implementation

### WIP

### About DDQN

[paper](https://arxiv.org/pdf/1509.06461)

### Environment

This time I used OpenAI's [Gym Retro](https://openai.com/index/gym-retro/) environment because it supports custom levels and allows direct access to RAM values, making it possible to inspect memory locations during gameplay.


