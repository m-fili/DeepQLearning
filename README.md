# Training Banana Collector using DQN

![](Images/Trained_Banana_Collector.gif)

## 1. Introduction
In this project, we'll train an agent using Deep Q-learning to collect as many yellow bananas 
as possible in a large square environment.

* __Reward and Punishment__: +1 reward for collecting yellow bananas, and -1 for blue bananas.
* __Goal__: Collect as many yellow bananas as possible and avoid blue bananas. This is an episodic
task and the gaol is to collect an average of +13 rewards in 100 consecutive tasks.
* __State Space__: The state space in this study has 37 dimensions, and contains agent's velocity
and ray-based perception of objects in front of the agent.
* __Action Space__: We have four possible actions anytime:
  * 0: Forward
  * 1: Backward 
  * 2: Left
  * 3: Right


## 2. Installation

### 2.1. Environment
First, you need to download the environment according to the Operating System:
* Linux: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OS: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows 64bit: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
* Windows 32bit: [[Download]](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, place the file in the root directory where the `Navigation.ipynb` exists. 


### 2.2. Python
For this project, you need _Python 3.9.x._ Old Anaconda version could be found 
[here](https://repo.anaconda.com/archive/).


### 2.3. Dependencies
You can download the packages needed using `requirements.txt` file:

```
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Training the Agent

The training procedure is shown in `Navigation.ipynb`. Follow this notebook to 
see how to train an agent to collect yellow bananas.

## 4. Visualizing the Agent
To Visualize the agent, you can use `VisualizeAgent.py` file.
















### Acknowledgement
This project is a part of
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
