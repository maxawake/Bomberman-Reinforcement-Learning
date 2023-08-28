import os
import pickle
import random
from .q_learning import COIN, CRATE
import numpy as np

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #if self.train or not os.path.isfile("my-saved-model.pt"):
    if not os.path.isfile("my-saved-crates-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = CRATE()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-crates-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.coin = False
    self.crate = True
    self.epsilon = 0
    self.start_epsilon_decaying = 1
    self.end_epsilon_decaying = 1000
    self.epsilon_decay_value = self.epsilon / (self.end_epsilon_decaying - self.start_epsilon_decaying)
    self.model.new_closest_coin_dist = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train:
        if random.uniform(0, 1) < self.epsilon:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(self.model.actions, p=[.1, .1, .1, .1, .1, .5])
        self.model.update()
        self.logger.debug("Querying model for action.")
        return self.model.actions[np.argmax(self.model.model[self.model.discrete_state_new])]
    else:
        if self.model.new_closest_coin_dist == 0:
            with open("my-saved-crates-model.pt", "rb") as file:
                self.model = pickle.load(file)
        else:
            if len(self.model.discrete_state_old) == 3:
                if self.model.discrete_state_old[2] > 8:
                    with open("my-saved-coins-model.pt", "rb") as file:
                        self.model = pickle.load(file)
        self.model.start_of_action(game_state)
        self.model.update()
        decision = self.model.actions[np.argmax(self.model.model[self.model.discrete_state_new])]
        self.model.end_of_action()
        return decision


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
