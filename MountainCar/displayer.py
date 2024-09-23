import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, axs = plt.subplots(2, figsize=(15, 15))


# loss
loss_file = open("log_MountainCar.txt", "r")
loss_lines = loss_file.read().split("\n")[:-1]
loss = pd.Series([float(l) for l in loss_lines])

axs[0].plot(loss.rolling(window=50).mean())
axs[0].grid()
axs[0].set_title("Loss in MountainCar-v0")

# rewards
rew_file = open("rew_MountainCar.txt", "r")
rew_lines = rew_file.read().split("\n")[:-1]
rewards = pd.Series([float(r) for r in rew_lines])

axs[1].plot(rewards.rolling(window=50).mean())
axs[1].grid()
axs[1].set_title("Rewards in MountainCar-v0")
plt.show()
