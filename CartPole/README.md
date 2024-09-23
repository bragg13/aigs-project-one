# Mini project one - Deep Reinforcement Learning

## Description

The project involves training an agent using Deep Reinforcement Learning so that it could play one of the games available in the Gym environment. Although I tried to use _MountainCar_ in the begininning as my environment, I eventually settled on _CartPole_ because I couldn't get any good results with MountainCar.

My code is subdivided in runnable cells:

- `imports and constants`, containing the imports we need and a set of constants for the MLP and DRL algorithms

- `MLP Network`, containing a neural network with two hidden layers, each with 128 neurons,and three functions:

  - `init_params_from_file` is used to create the network loading saved parameters from the _weights.pkl_ file for the play part;
  - `init_params` is used to create the network initialising the weights and biases randomly;
  - `forward` is the forward step of the NN

- `Policies`, containing the two policies used by the agent:

  - `random_policy` to return a random action;
  - `policy` to return a random action or the best action available, based on the value of Epsilon (exploration/exploitation)

- `Loss function`, where we calculate the loss using mean squared error between the predicted values from the Q-network and the target values calculated using the Bellman equation from the Target-network

- `Training step`, where the gradients of the loss function are calculated using _jax_, and the updated parameters of the Q-network are calculated

- `Playing`, where the Q-network is initialised with trained parameters in `weights.pkl`, then the simulation is launched, with the agent always selecting the best action

- `Training`, where we initialise the Q-net randomly, and then get into a training loop. We choose an action using the policy, and give additional reward based on how the game finished.
  After some number of episodes, we update the target network, and/or decay epsilon.
  When we have some number of batches in memory, we sample after an episode and perform the training step for the Q-net.
  We log rewards and losses and save the weights to file.

## Results

The agent learned to beat the game to a good extent. The loss and reward trends are available in `training_trends.png`.
Running it 100 times, it gets an average reward of 379.37 with std 95.92. The unstability might be due to the random initialisation of Gym environment. A plot is available in `rewards_playing.png`.

## Components needed

[x] MLP network: observation as input, q-values for each action as output
[x] agent: selects actions based on q-values
[x] memory buffer: stores transitions used for training
[x] training loop: samples from the buffer and updates the NN params

## Files:

- .py script that trains the agent
- .md file to explain
- .pkl with trained parameters
- .gif of my agent playing the game
- .gif of random agent playing the game
