# =================
# %% imports
# =================
import gymnasium as gym
import jax.numpy as jnp
from collections import deque, namedtuple
from jax import nn, random
import random as py_rand
import jax
import sys
from tqdm import tqdm

# =================
# %% constants
# =================
if sys.argv and len(sys.argv) > 1:
    TRAINING = True if sys.argv[1] == "True" else False
else:
    TRAINING = False
MAX_MEMORY_LEN = 10000
BATCH_SIZE = 64
MIN_BATCH = 100
TARGET_NET_UPDATE_FREQ = 50
GAMMA = 0.998
LEARNING_RATE = 5e-4
EPSILON_DECAY = 0.01
EPSILON_MIN = 0.8
NUM_EPISODES = 100000
GAME = "MountainCar"


# =================
# %% MLP Network
# =================
def mlp():
    hidden_neurons = 128

    def init_params_from_file(filename):
        wf = open(filename, "rb")
        saved_nn = jnp.load(wf)
        params = {
            "w1": saved_nn["w1"],
            "w2": saved_nn["w2"],
            "w3": saved_nn["w3"],
            "b1": saved_nn["b1"],
            "b2": saved_nn["b2"],
            "b3": saved_nn["b3"],
        }
        wf.close()
        return params

    def init_params(key, state_dimension, action_dimension):
        return {
            "w1": random.normal(key, (state_dimension, hidden_neurons)) * 0.01,
            "b1": jnp.zeros(hidden_neurons) * 0.01,
            "w2": random.normal(key, (hidden_neurons, hidden_neurons)) * 0.01,
            "b2": jnp.zeros(hidden_neurons) * 0.01,
            "w3": random.normal(key, (hidden_neurons, action_dimension)) * 0.01,
            "b3": jnp.zeros(action_dimension) * 0.01,
        }

    def forward(params, x_data):
        z = x_data @ params["w1"] + params["b1"]
        z = nn.relu(z)
        z = z @ params["w2"] + params["b2"]
        z = nn.relu(z)
        z = z @ params["w3"] + params["b3"]
        return z

    return init_params_from_file, init_params, forward


# ====================
# %% Agent (policies)
# ====================
def random_policy(rng, action_space):
    n = action_space.__dict__["n"]
    action = random.randint(rng, (1,), 0, n).item()
    return action


def policy(rng, params, state, EPSILON, action_space=None):
    # explore/exploit
    choice = py_rand.random()
    if choice <= EPSILON:
        return random_policy(rng, action_space)
    else:
        qvalues = forward(params, state)
        action = jnp.argmax(qvalues)
        return int(action)


# ====================
# %% Loss function
# ====================
def loss_fn(params, target_params, batch):
    states, actions, rews, next_states, is_dones = batch
    # calculate and select the predicted action values
    # - I pass the current state(s) to the Q-net, using the q-params
    # and I get the predicted Q-value for all the actions
    predicted = forward(params, states)

    # I select the Q-values corresponding to the actions taken
    selected_qval = predicted[jnp.arange(len(actions)), actions]

    # now calculate the Q-values for the next state(s) with the bellmann equation
    # and using the target-net with target-params
    # Q = r + gamma * maxQ'(s', a') * (1-is_done)

    # calculate the target action values
    next_qval = forward(target_params, next_states)
    max_next_qval = jnp.max(next_qval, axis=1)

    targets = rews + GAMMA * max_next_qval * (1 - is_dones)

    # calculate loss using mean squared error
    loss = jnp.mean((targets - selected_qval) ** 2)
    return loss


# ====================
# %% Training step
# ====================
@jax.jit
def train(params, target_params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, target_params, batch)

    # aggiorno i parametri della Q-net
    w1 = params["w1"] - LEARNING_RATE * grads["w1"]
    w2 = params["w2"] - LEARNING_RATE * grads["w2"]
    w3 = params["w3"] - LEARNING_RATE * grads["w3"]
    b1 = params["b1"] - LEARNING_RATE * grads["b1"]
    b2 = params["b2"] - LEARNING_RATE * grads["b2"]
    b3 = params["b3"] - LEARNING_RATE * grads["b3"]

    params = {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "b1": b1,
        "b2": b2,
        "b3": b3,
    }

    return params, loss


# ====================
# %% Playing
# ====================
def do_play(EPSILON):
    logf = open(f"rewards_playing_{GAME}.txt", "w")
    env = gym.make("MountainCar-v0", render_mode="human")

    trained_params = init_params_from_file(f"weights_{GAME}.pkl")
    print("loaded weights: ", {k: v.shape for k, v in trained_params.items()})

    for i in range(100):
        state, info = env.reset()
        done = False
        total_rew = 0
        step = 0

        while not done:
            action = policy(rng, trained_params, state, EPSILON)
            # action = random_policy(rng, env.action_space) # for random agent playing
            next_state, rew, is_terminated, is_truncated, info = env.step(action)
            total_rew += float(rew)

            # print(f"step {step} act: {action}")
            # print(f"rew {rew} total rew {total_rew}")

            if is_terminated or is_truncated:
                done = True

            state = next_state
            step += 1

        print(f"episode {i} ended after {step} steps, with total reward of {total_rew}")
        logf.write(str(total_rew))
        logf.write("\n")
        logf.flush()

    logf.close()
    env.close()


# ====================
# %% Training
# ====================
def parse_state(next_state):
    # position
    pos = round(next_state[0], 1) * 10

    # velocity
    vel = round(next_state[1], 2) * 100

    pos = int(pos)
    vel = int(vel)
    return jnp.array([pos, vel])


def do_train(EPSILON):
    # logging
    logf = open(f"log_{GAME}.txt", "w")
    rewf = open(f"rew_{GAME}.txt", "w")
    losses = []
    rews = []

    env = gym.make("MountainCar-v0")

    # for experience replay
    entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])
    memory = deque(maxlen=1000)

    # NN init
    qnet_params = init_params(
        rng, env.observation_space.shape[0], env.action_space.__dict__["n"]
    )
    target_params = qnet_params

    for episode in tqdm(range(NUM_EPISODES)):
        episode_done = False
        ep_losses = []
        ep_rews = []
        state, info = env.reset()
        state = parse_state(state)

        # training loop
        while not episode_done:
            action = policy(rng, qnet_params, state, EPSILON, env.action_space)

            # step next episode
            next_state, rew, is_terminated, is_truncated, info = env.step(action)

            # additional reward
            if GAME == "CartPole":
                if is_terminated:
                    rew = -200
                if is_truncated:
                    rew = 100
            elif GAME == "MountainCar":
                if is_terminated:
                    rew = 200
                if is_truncated:
                    rew = -100

                next_state = parse_state(next_state)

                # rew += 1000 * next_state[1]

            # store the experience in the replay buffer
            ep_rews.append(rew)
            _entry = entry(state, action, rew, next_state, is_terminated | is_truncated)
            memory.append(_entry)

            state = next_state

            # sometimes update the target network with the values from qnet
            if episode % TARGET_NET_UPDATE_FREQ == 0:
                target_params = qnet_params

            # decay epsilon every 100 ep
            if episode % 100 == 0:
                EPSILON = max(EPSILON - EPSILON_DECAY, EPSILON_MIN)

            # episode is done if truncated or terminated
            if is_terminated or is_truncated:
                episode_done = True

        # perform actual training only at the end of the episode
        if len(memory) >= MIN_BATCH:
            # sample a mini batch :D
            batch = py_rand.sample(memory, BATCH_SIZE)
            batch = tuple(map(jnp.array, zip(*batch)))

            # perform training step
            updated_params, loss = train(qnet_params, target_params, batch)
            qnet_params = updated_params

            ep_losses.append(loss)

        # logging
        _loss = jnp.array(ep_losses).mean()
        _rew = jnp.array(ep_rews).mean()
        losses.append(_loss)
        rews.append(_rew)

        logf.write(str(_loss))
        logf.write("\n")
        logf.flush()

        rewf.write(str(_rew))
        rewf.write("\n")
        rewf.flush()

        # break when reached some good performance
        # if len(losses) > 1000 and min(losses) == _loss:
        #     break

    # save trained weights
    with open(f"weights_{GAME}.pkl", "wb") as wf:
        jnp.savez(
            wf,
            w1=qnet_params["w1"],
            w2=qnet_params["w2"],
            w3=qnet_params["w3"],
            b1=qnet_params["b1"],
            b2=qnet_params["b2"],
            b3=qnet_params["b3"],
        )
    logf.close()
    rewf.close()
    env.close()


# ====================
# %% Main
# ====================
# generic stuff

rng = random.PRNGKey(0)
EPSILON = 1.0 if TRAINING else 0.0

# network
init_params_from_file, init_params, forward = mlp()

if not TRAINING:
    do_play(EPSILON)
else:
    do_train(EPSILON)
