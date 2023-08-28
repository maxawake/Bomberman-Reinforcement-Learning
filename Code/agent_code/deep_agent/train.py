import random
from collections import namedtuple, deque
from typing import List
import torch
import numpy as np
from .callbacks import state_to_features, calc_true_dist, get_closest_element, get_bomb_zone, get_danger_zone
from .DQN import device, optimize_model, action_to_number
from torch.utils.tensorboard import SummaryWriter
from .params import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

WAITED_NO_BOMB = "WAITED_NO_BOMB"
LOOP = "LOOP"
WALL_AREA = "WALL_AREA"
REPEAT = "REPEAT"
CORNER = "CORENR"
CLOSER_COIN = "CLOSTER_COIN"
LAST_COIN = "LAST_COIN"
BACKSTEP = "BACKSTEP"
RIGHT_COIN = "RIGHT_COIN"
RUN_AWAY = "RUN_AWAY"
CLOSER_CRATE = "CLOSER_CRATE"
DANGER_ZONE = "DANGER_ZONE"
PERFECT_MOVE = "PERFECT_MOVE"
WRONG_BOMB = "WRONG_BOMB"
FURTHER_CRATE = "FURTHER_CRATE"
CLOSER_BOMB = "CLOSER_BOMB"


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a random back of size batch_size from memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def setup_training(self):
    """Initialize training"""
    print("Start training...")
    #self.writer = SummaryWriter()
    self.memory = ReplayMemory(MEMORY_SIZE)
    self.coordinate_history = deque([], 20)
    self.loops = deque([], LOOPS_SIZE)
    self.danger_distance = 0
    self.steps_done = 0
    self.coin_distance = 100
    self.crate_distance = 100
    self.round = 0
    self.reward_sum = []
    self.q_values_sum = []
    self.highest_reward = 0


def pack_states(self, old_state, action, new_state, events):
    """Packs the states, action and reward as tensors for batch training"""
    ogs = state_to_features(self, old_state)
    action_number = torch.tensor(action_to_number(action), device=device, dtype=torch.long).unsqueeze(0)
    if new_state is not None:
        ngs = state_to_features(self, new_state)
    else:
        ngs = None
    reward = torch.tensor(reward_from_events(self, events), device=device, dtype=torch.long).unsqueeze(0)
    return ogs, action_number, ngs, reward


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.round = new_game_state["round"]
    self.step = new_game_state["step"]
    field = new_game_state["field"]
    self.steps_done += 1
    name, score, bombs_left, (x, y) = new_game_state['self']
    self.score = score
    bombs = new_game_state['bombs']
    bombs = np.array([xy for (xy, t) in bombs])
    coins = new_game_state['coins']
    position = np.array([x, y])
    crates = np.argwhere(field == 1)

    if VERBOSE:
        print("Round", self.round)
        print("Step", self.step)
        print("Position", x, y)
        print("coins", coins)

    # Checks if position has been seen more than 2 times in the last 20 steps
    if self.coordinate_history.count((x, y)) > 4:
        events.append(LOOP)
    self.coordinate_history.append((x, y))

    if len(coins) != 0:
        # Calculate new distance to the last closest coin and compare to old distance
        new_distance, _ = calc_true_dist(self.closest_coin, position)
        if new_distance < self.coin_distance and self.accs_bool and not self.danger_zone:
            events.append(CLOSER_COIN)
        self.coin_distance = new_distance

    # If there are crates, give reward for stepping closer to the closest crate
    if len(crates) != 0:
        new_distance, _ = calc_true_dist(self.closest_crate, position)
        if new_distance < self.crate_distance and not self.danger_zone:
            events.append(CLOSER_CRATE)
        if new_distance >= self.crate_distance and not self.danger_zone:
            events.append(FURTHER_CRATE)
        self.crate_distance = new_distance

        # Give special reward if bomb was dropped directly infront of a crate
        if e.BOMB_DROPPED in events and get_bomb_zone(self.relative_crate) == 1:
            events.append(PERFECT_MOVE)

    # if there are bombs, give reward for running away from the closest bomb
    if len(bombs) != 0:
        bomb, rel, new_distance = get_closest_element(bombs, position)
        if new_distance > self.danger_distance:
            events.append(RUN_AWAY)
        if new_distance <= self.danger_distance:
            events.append(CLOSER_BOMB)
        self.danger_distance = new_distance
        # Give negative reward for beeing in a danger zone
        if get_danger_zone(bombs, position) == 1 and e.WAITED in events:
            events.append(DANGER_ZONE)
    else:
        # Give negative reward for waiting when there is no surrounding bomb
        if e.WAITED in events:
            events.append(WAITED_NO_BOMB)

    # Give penalty for dropping a bomb when there is no way out
    if self.no_way and e.BOMB_DROPPED in events:
        events.append(WRONG_BOMB)

    if self_action is not None:
        # Pack and push the experience in the memory
        if not self.coin_collector:
            ogs, action, ngs, reward = pack_states(self, old_game_state, self_action, new_game_state, events)
            self.memory.push(ogs, action, ngs, reward)

        # Update the policy network
        optimize_model(self)

        # Update the target network, copying all weights and biases in DQN
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Pack and push the experience in the memory
    ogs, action, ngs, reward = pack_states(self, last_game_state, last_action, None, events)
    self.memory.push(ogs, action, ngs, reward)

    # Write statistics to tensorboard
    if self.round > 1:
        mean_reward = np.mean(self.reward_sum)
        mean_qvalues = np.mean(self.q_values_sum)
        #self.writer.add_scalar('Average Reward', mean_reward, self.round)
        #self.writer.add_scalar('Average Q-Values', mean_qvalues, self.round)

        if VERBOSE:
            print("Mean reward", mean_reward)
            print("Average Q-Values", mean_qvalues)

        if mean_reward > self.highest_reward:
            # Save state
            state = {
                'epoch': self.round,
                'policy_net': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(state, "models/best_model.pt")
            self.highest_reward = mean_reward

    self.reward_sum = []
    self.q_values_sum = []

    # Optimize last time for round
    optimize_model(self)

    # Save state
    if self.round % 10 == 0:
        state = {
            'epoch': self.round,
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, MODEL_PATH)


def reward_from_events(self, events: List[str]) -> int:
    reward_sum = 0

    game_rewards = {
        FURTHER_CRATE: -10,
        CLOSER_CRATE: 10,
        LOOP: -5,
        WAITED_NO_BOMB: -5,
        DANGER_ZONE: -10,
        WRONG_BOMB: -10,
        PERFECT_MOVE: 40,
        RUN_AWAY: 25,
        CLOSER_BOMB: -15,
        e.INVALID_ACTION: -3,
        e.BOMB_DROPPED: -20,
        e.CRATE_DESTROYED: 5,
    }

    # Assign rewards for events
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # Save reward sum for statistics
    self.reward_sum.append(reward_sum)
    if VERBOSE:
        print("Events and reward:", events, reward_sum, self.score)
    #print(events, reward_sum)
    return reward_sum
