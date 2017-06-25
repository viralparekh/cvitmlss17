# cvitmlss17
http://cvit.iiit.ac.in/mlsummerschool2017/

## Lab 4: Deep Reinforcement Learning
Reinforcement learning provides the capacity for us not only to teach an artificial agent how to act, but to allow it to learn through it's own interaction with the environment. By combining the complex representations that deep neural networks can learn with the goal-driven learning of an RL agent, computers have accompolished some amazing feats, like beating humans in [dozen Atari games](https://deepmind.com/research/dqn/), and [defeating the Go world champion](https://deepmind.com/research/alphago/).

In this lab, we will explore the two paradigms of RL algorithms, namely, the policy-based algorithms and the Q-Learning algorithms. The detailed [^1] plan is as follows:

+ **_Policy Based Methods_**
    1. **Simple Policy**: An implementation of policy gradient method for stateless environments such as n-armed bandit problem. 
    2. **Contextual Policy**: An implementation of policy gradient method for stateful environments such as contextual bandit problem.
    3. **Policy Network**: An implementation of a neural network policy-gradient agent that solves full RL problems with states and delayed rewards, and two opposite actions (ie. Cartpole and/or Pong).
    4. **Vanilla Policy**[^2]: An implementation of a neural network vanilla-policy-gradient agent that solves full RL problem with states, delayed rewards, and an arbitrary number of actions.
    5. **Model Network**[^3]: An addition to the policy network algorithm which includes a separate network which models the environment dynamics.

+ **_Q-Learning Algorithms_**
    1. **Q-Table**: An implementation of Q-Learning using tables to solve stochastic environment problem.
    2. **Q-Network**: A simple neural network implementation of Q-Learning to solve the same environment as in Q-Table.
    3. **Deep Q-Network and Double-Duelling DQN**[^2]: An implementation of DQN with the Double DQN and Duelling DQN additions to improve stability and performance.
    4. **Q-Exploration**[^2]: An implementation of DQN containing multiple action-selection strategies for exploration.
    5. **Deep Recurrent Q Network**[^3]: An implementation of Deep Recurrent Q-Network which can solve reinforcement learning problems involving partial observability.

[^1]: This is the initial tentative plan. This might get updated/changed later on.
[^2]: These will be given as programming assignment.
[^3]: These are optional advanced reading/coding excercise.

