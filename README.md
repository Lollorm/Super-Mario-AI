# Super Mario Bros AI

# CURRENTLY WIP

![mario1-1](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/mario1-1.gif)

An AI agent trained with the NeuroEvolution of Augmenting Topologies (NEAT) algorithm successfully beating Level 1-1  
(Even going as far as learning a frame/pixel perfect glitch called 'wall jump')

![Best fitness over generations](https://github.com/Lollorm/Super-Mario-Bros-AI/blob/main/NEAT%20Neuroevolution%20of%20Augmenting%20Topologies/First%20agent%20that%20managed%20to%20beat%201-1/fitness_progress_0_to_350.png)

Best fitness over generations

## Environment

The Gym Super Mario Bros environment was used to provide the game interface and training environment for the agent. Documentation can be found [here](https://pypi.org/project/gym-super-mario-bros/).

```
pip install gym-super-mario-bros
```

### To play manually:

```
gym_super_mario_bros -e 'SuperMarioBrosRandomStages-v0' -m 'human' --stages '1-4'
```

## NEAT implementation

### About NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm used to evolve artificial neural networks. It gradually optimizes both network structure and weights through evolutionary processes. A full description of the algorithm is available in the original [paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).  
In this project, NEAT is implemented using the NEAT-Python library. The official documentation can be found [here](https://neat-python.readthedocs.io/en/latest/).


### Requirements

```
pip install gym==0.25.1 gym-super-mario-bros==7.4.0 nes-py==8.2.1 neat-python==1.1.0 numpy==1.26.4 opencv-python==4.12.0.88
```


work in progress

## DDQN implementation

### WIP

### About DDQN

![paper](https://arxiv.org/pdf/1509.06461)

