import torch
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from .DQN import select_action, device, load_model, load_coin_net
from .params import *


def setup(self):
    load_model(self, MODEL_PATH)
    load_coin_net(self, COIN_PATH)
    initialize_variables(self)


def initialize_variables(self):
    self.count = 0
    self.coin_count = 0
    self.steps_done = 0
    self.danger_zone = False
    self.last_action = None
    self.accs_bool = False
    self.no_way = False
    self.coin_collector = False
    self.size_crates = 0
    self.closest_crate = np.array([0, 0])
    self.closest_coin = np.array([0, 0])
    self.crate_distance = 0


def act(self, game_state: dict) -> str:
    if not self.train:
        coins = np.array(game_state['coins'])
        field = np.array(game_state['field'])
        pos = np.array(game_state["self"][3])
        bombs = game_state["bombs"]
        bombs = np.array([xy for (xy, t) in bombs])
        if len(coins) != 0:# and get_danger_zone_reloaded(bombs, pos) == 0:
            self.accs = get_accessible(field, pos, coins)
            if 8 > len(self.accs) > 0:
                path_len = get_distance(field, pos, self.accs[0,0], True)
                with torch.no_grad():
                    q_values = self.coin_net(state_to_features(self, game_state))
                    self.coin_collector = True
                    return ACTIONS[q_values.max(1)[1].view(1, 1)]

    self.last_action = ACTIONS[select_action(self, state_to_features(self, game_state))]
    self.coin_collector = False
    if VERBOSE:
        print("Action: ", self.last_action)
    return self.last_action


def get_distance(field, start, end, path_=False):
    field = np.where(field != 0, -1, 1).T
    grid = Grid(matrix=field)
    start = grid.node(start[0], start[1])
    end = grid.node(end[0], end[1])
    finder = AStarFinder()
    path, runs = finder.find_path(start, end, grid)
    # print("hello", grid.grid_str(path=path, start=start, end=end), field)
    if path_:
        if len(path) <= 1:
            return [0, 0]
        else:
            return path
    else:
        return len(path)


def calc_true_dist(element, position):
    """Correts the manhatten-distance by +2 to account for walls"""
    elem = np.array(element)
    relative = elem - position
    dist = np.sum(np.abs(relative))
    if position[0] % 2 == 0 and position[1] != 0 and relative[0] == 0 and relative[1] != 0:
        dist = dist + 2
    if position[1] % 2 == 0 and position[0] != 0 and relative[1] == 0 and relative[0] != 0:
        dist = dist + 2
    return dist, relative


def get_closest_element(elements, position):
    """Get all corrected distances and return the closest of all elements"""
    distance, relative = [], []
    for elem in elements:
        dist, rel = calc_true_dist(elem, position)
        distance.append(dist)
        relative.append(rel)
    relative = np.array(relative)
    distance = np.array(distance)
    idx = np.argmin(distance)
    return elements[idx], relative[idx], distance[idx]


def get_bomb_zone(rel_crate):
    """Returns 1 if the closest crate is directly beneath the agent"""
    if rel_crate[0] == 0 and np.abs(rel_crate[1]) == 1 or rel_crate[1] == 0 and np.abs(rel_crate[0]) == 1:
        return 1
    else:
        return 0


def get_bombs(bombs, pos, features):
    """Get the direction of the closest bomb"""
    relative = bombs - pos
    for bomb in relative:
        if bomb[0] >= 0:
            features[12] = 1
        if bomb[0] <= 0:
            features[13] = 1
        if bomb[1] >= 0:
            features[14] = 1
        if bomb[1] <= 0:
            features[15] = 1
    return features


def get_accessible(field, position, coins):
    accs = np.array([get_distance(field, position, coin) for coin in coins])
    accs = np.argwhere(accs != 0)
    return coins[accs]


def get_danger_zone(bombs, pos):
    """Return 1 if agent is in the destruction zone of the bombs"""
    if len(bombs) != 0:
        relative = np.abs(bombs - pos)
        for i, bomb in enumerate(bombs):
            if bomb[0] == pos[0] and relative[i][1] < 4 or bomb[1] == pos[1] and relative[i][0] < 4:
                return 1
        return 1
    else:
        return 0


def get_danger_zone_reloaded(bombs, position):
    """
    gives the featback if the players state is a state of danger,
     and if where the danger (bomb) is relative to player, or not (range of bomb or not)
    Inputs:
        state: game_state, get_relative_bomb(game_state)
    Returns:
    index_danger_zone in update
    """
    if len(bombs) == 0:
        # no bombs laid yet
        return 0
    else:
        danger_zone = []
        for bomb in bombs:
            if bomb[1] % 2 == 0:
                for i in range(-3, 4):
                    danger_zone.append([bomb[0], bomb[1]+i])
            elif bomb[0] % 2 == 0:
                for i in range(-3, 4):
                    danger_zone.append([bomb[0]+i, bomb[1]])
            else:
                for i in range(-3, 4):
                    danger_zone.append([bomb[0]+i, bomb[1]])
                    danger_zone.append([bomb[0], bomb[1]+i])

        if list(position) in danger_zone:
            return 1
        else:
            return 0


def get_free_way(game_state, position, features, distance_bomb=0, relative_bomb=None):
    x = int(position[0])
    y = int(position[1])

    field = game_state["field"]
    bombs = game_state["bombs"]
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]

    bomb_map = np.ones(field.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    if distance_bomb != 0:
        features[6:10] = 1

        # Check which moves make sense at all
        directions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        valid_tiles = []
        for d in directions:
            if ((field[d] == 0) and
                    (game_state['explosion_map'][d] <= 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)

        if (x - 1, y) in valid_tiles:
            features[6] = 0
        if (x + 1, y) in valid_tiles:
            features[7] = 0
        if (x, y - 1) in valid_tiles:
            features[8] = 0
        if (x, y + 1) in valid_tiles:
            features[9] = 0

        if relative_bomb[0] < 0:
            features[6] = 1
        if relative_bomb[0] > 0:
            features[7] = 1
        if relative_bomb[1] < 0:
            features[8] = 1
        if relative_bomb[1] > 0:
            features[9] = 1

    else:
        field = np.abs(field)
        if x % 2 == 1 and y % 2 == 1:
            # links
            if x != 1:
                u = field[x - 1, y]
                uu = field[x - 2, y - 1:y + 2]
                if u == 1 or uu[1] == 1 or (uu[0] == 1 and uu[2] == 1):
                    features[6] = 1
            else:
                features[6] = 1
            # rechts
            if x != 15:
                d = field[x + 1, y]
                dd = field[x + 2, y - 1:y + 2]
                if d == 1 or dd[1] == 1 or (dd[0] == 1 and dd[2] == 1):
                    features[7] = 1
            else:
                features[7] = 1
            # oben
            if y != 1:
                l = field[x, y - 1]
                ll = field[x - 1:x + 2, y - 2]
                if l == 1 or ll[1] == 1 or (ll[0] == 1 and ll[2] == 1):
                    features[8] = 1
            else:
                features[8] = 1
            # unten
            if y != 15:
                r = field[x, y + 1]
                rr = field[x - 1:x + 2, y + 2]
                if r == 1 or rr[1] == 1 or (rr[0] == 1 and rr[2] == 1):
                    features[9] = 1
            else:
                features[9] = 1

        else:
            u = field[x - 1, y - 1:y + 2]
            if u[1] == 1 or (u[0] == 1 and u[2] == 1):
                features[6] = 1

            d = field[x + 1, y - 1:y + 2]
            if d[1] == 1 or (d[0] == 1 and d[2] == 1):
                features[7] = 1

            l = field[x - 1:x + 2, y - 1]
            if l[1] == 1 or (l[0] == 1 and l[2] == 1):
                features[8] = 1

            r = field[x - 1:x + 2, y + 1]
            if r[1] == 1 or (r[0] == 1 and r[2] == 1):
                features[9] = 1

        if x < 13:
            ulong = field[x + 1:x + 5, y]
            # rechts
            if np.all(ulong == 0) or np.all(ulong[:-1] == 0) and field[x + 3, y - 1] == 0 and field[x + 3, y + 1] == 0:
                features[7] = 0
        if x > 3:
            dlong = field[x - 4:x, y]
            # links
            if np.all(dlong == 0) or np.all(dlong[:-1] == 0) and field[x - 3, y - 1] == 0 and field[x - 3, y + 1] == 0:
                features[6] = 0
        if y < 13:
            llong = field[x, y + 1:y + 5]
            # unten
            if np.all(llong == 0) or np.all(llong[:-1] == 0) and field[x + 1, y + 3] == 0 and field[x - 1, y + 3] == 0:
                features[9] = 0
        if y > 3:
            rlong = field[x, y - 4:y]
            # oben
            if np.all(rlong == 0) or np.all(rlong[:-1] == 0) and field[x + 1, y - 3] == 0 and field[x - 1, y - 3] == 0:
                features[8] = 0

    return features


def state_to_features(self, game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    if game_state["step"] == 1:
        initialize_variables(self)

    # Initialize an empty matrix, to be filled with the environment features
    features = np.zeros(FEATURE_SIZE)
    self.danger_zone = False
    # Get important environment variables
    pos = np.array(game_state["self"][3])
    bombs = game_state['bombs']
    others = game_state['others']
    bombs = np.array([xy for (xy, t) in bombs])
    others = np.array([xy for (n, s, b, xy) in others])

    coins = np.array(game_state['coins'])
    field = np.array(game_state['field'])
    crates = np.argwhere(field == 1)

    # Treat others as crates
    if len(others) != 0:
        crates = np.vstack((crates, others))

    # if there are crates, target closest crate of the moment
    if len(crates) != 0:
        self.closest_crate, self.relative_crate, self.crate_distance = get_closest_element(crates, pos)
        direction = self.relative_crate

        features[10] = get_bomb_zone(self.relative_crate)
        if direction[0] >= 0:
            features[0] = 1
        if direction[0] <= 0:
            features[1] = 1
        if direction[1] >= 0:
            features[2] = 1
        if direction[1] <= 0:
            features[3] = 1

    if len(coins) != 0:
        self.accs = get_accessible(field, pos, coins)
        if self.closest_coin.tolist() not in coins.tolist() or game_state["self"] == 1:
            self.closest_coin, self.relative_coin, self.coin_distance = get_closest_element(coins, pos)
        else:
            self.relative_coin = calc_true_dist(self.closest_coin, pos)[1]
        if self.closest_coin in self.accs:
            features[20] = 1
            self.accs_bool = True
            direction = np.array(get_distance(field, pos, self.closest_coin, True)[1]) - pos
            if direction[0] >= 0:
                features[16] = 1
            if direction[0] <= 0:
                features[17] = 1
            if direction[1] >= 0:
                features[18] = 1
            if direction[1] <= 0:
                features[19] = 1
        else:
            self.accs_bool = False

    features[4] = pos[0] % 2
    features[5] = pos[1] % 2

    if bombs.size != 0:
        accs = get_accessible(field, pos, bombs)
        closest_bomb, relative_bomb, distance_bomb = get_closest_element(bombs, pos)
        if closest_bomb in accs and distance_bomb < 8:
            features = get_bombs([closest_bomb], pos, features)
            features = get_free_way(game_state, pos, features, distance_bomb, relative_bomb)
            features[11] = get_danger_zone(bombs, pos)
            if features[11] == 1:
                self.danger_zone = True
            else:
                self.danger_zone = False

    if np.all(features[6:10] == 1):
        self.no_way = True
    else:
        self.no_way = False
    return torch.tensor(features, device=device, dtype=torch.float).unsqueeze(0)
