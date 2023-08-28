import math
import random
import numpy as np
import os
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .params import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# if gpu is to be used
device = torch.device("cpu")
print("Device: ", device)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_size, 64)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(64, 64)
        self.relu2 = nn.LeakyReLU()
        self.l3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x.view(x.size(0), -1)


def load_model(self, path):
    if RANDOM_SEED is not None:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    self.policy_net = DQN(FEATURE_SIZE, 6).to(device)
    self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

    if os.path.isfile(path):

        state = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(state["policy_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        print("Model: {}".format(path))
        if not self.train:
            self.policy_net.eval()
            print("Evaluated model for inference")
    else:
        print("Initializing new model")

    self.target_net = DQN(FEATURE_SIZE, 6).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()


def load_coin_net(self, path):
    self.coin_net = DQN(FEATURE_SIZE, 6).to(device)
    if os.path.isfile(path):
        state = torch.load(path, map_location=device)
        self.coin_net.load_state_dict(state["policy_net"])
        self.coin_net.eval()


def action_to_number(action):
    """Returns number according to chosen action"""
    for i, a in enumerate(ACTIONS):
        if a == action:
            return i


def select_action(self, state):
    sample = random.random()
    with torch.no_grad():
        q_values = self.policy_net(state)
        if self.train:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * self.round / EPS_DECAY)
            TEMP = (TEMP_END - TEMP_START) / TEMP_DECAY * self.round + TEMP_START
            self.q_values_sum.append(q_values.squeeze(0).cpu().numpy())
        else:
            eps_threshold = EPS_PLAY
        if sample > eps_threshold:
            return q_values.max(1)[1].view(1, 1)
        else:
            # Max-Boltzmann exploration
            if MODE == "BOLTZMANN":
                q_values = q_values.detach().cpu().numpy()
                propabilities = np.exp(q_values / TEMP) / np.sum(np.exp(q_values / TEMP))
                return torch.tensor([[np.random.choice(np.arange(0, 6), p=propabilities[0])]], device=device,
                                    dtype=torch.long)
            if MODE == "EPSILON":
                return torch.tensor([[np.random.choice(np.arange(0, 6), p=[0.1,0.1,0.1,0.1,0.1,0.5])]], device=device, dtype=torch.long)


def optimize_model(self):
    if len(self.memory) < BATCH_SIZE:
        return
    transitions = self.memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).reshape(BATCH_SIZE, 1)
    reward_batch = torch.cat(batch.reward)

    q_values = self.policy_net(state_batch).gather(1, action_batch)

    target_pred = torch.zeros(BATCH_SIZE, device=device)
    target_pred[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    target_values = (target_pred * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()

    # Clamp the gradients if necessary
    if CLAMPED:
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

    self.optimizer.step()
