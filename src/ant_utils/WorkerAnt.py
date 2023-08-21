import copy
import random

import numpy as np

from .Ant import Ant

surrounding_to_move = {
    'right': 'increase_x',
    'left': 'decrease_x',

    'forward': 'decrease_z',
    'backward': 'increase_z',

    'up': 'climb_up',
    'down': 'climb_down'
}


def invert(move: str):
    if move == 'increase_x':
        return 'decrease_x'
    if move == 'decrease_x':
        return 'increase_x'

    if move == 'increase_z':
        return 'decrease_z'
    if move == 'decrease_z':
        return 'increase_z'

    if move == 'climb_up':
        return 'climb_down'
    if move == 'climb_down':
        return 'climb_up'

    if move == 'pick_item':
        return 'release_item'

    return 'pick_item'


def lower_probs(probs: dict, moves: list, final_value: float = 0.01):
    if len(list(probs.keys())) == len(moves):
        return probs

    to_lower_keys = list()

    for move in moves:
        if move in probs and probs[move] > final_value:
            to_lower_keys.append(move)

    if len(to_lower_keys) == 0:
        return probs

    to_redistribute = 0
    for move in to_lower_keys:
        if move in probs:
            to_redistribute += (probs[move] - final_value)

    keys_to_increase = len(list(probs.keys())) - len(moves)
    prob_increase = to_redistribute / keys_to_increase
    for key in list(probs.keys()):
        if key in moves and key not in to_lower_keys:
            continue

        if key not in to_lower_keys:
            probs[key] += prob_increase
        else:
            probs[key] = final_value

    return probs


def increase_probs(probs: dict, moves: list, final_value: float = 0.7):
    if len(list(probs.keys())) == len(moves):
        return probs

    to_increase_keys = list()

    for key in probs:
        if probs[key] < final_value and key in moves:
            to_increase_keys.append(key)

    diff = final_value
    for key in to_increase_keys:
        if key in moves:
            diff -= probs[key]

    value_to_add = diff / len(moves)
    value_to_remove = diff / (len(list(probs.keys())) - len(moves))

    for key in probs:
        if key in moves and not key in to_increase_keys:
            continue
        if key in to_increase_keys:
            probs[key] = probs[key] + value_to_add
        else:
            probs[key] = probs[key] - value_to_remove

    return probs


def increase_probs_with_pheromones(probs: dict, moves: list, increase_by: float = 0.5):
    if len(list(probs.keys())) == len(moves):
        return probs

    diff_to_add = increase_by / len(moves)
    diff_to_remove = increase_by / (len(list(probs.keys())) - len(moves))

    for key in probs:
        if key in moves:
            probs[key] = probs[key] + diff_to_add
        else:
            probs[key] = probs[key] + diff_to_remove

    return probs


def find_move(position_1: tuple, position_2: tuple):
    # Case in which the ant has picked up or released an item without being able to move due to map limitations
    if position_1 == position_2:
        return 0

    if position_1[0] - position_2[0] == 1:
        return 'decrease_x'
    if position_1[0] - position_2[0] == -1:
        return 'increase_x'

    if position_1[2] - position_2[2] == 1:
        return 'decrease_z'
    if position_1[2] - position_2[2] == -1:
        return 'increase_z'

    if position_1[1] - position_2[1] == 1:
        return 'climb_up'
    if position_1[1] - position_2[1] == -1:
        return 'climb_down'

    return None


class WorkerAnt(Ant):
    def __init__(self, **kwargs):
        super().__init__(type='worker', **kwargs)
        self.spawn_position = self.position
        self.id = kwargs['id']
        self.cycles_from = np.array(list(), dtype=np.uint8)
        self.path = list()

        self.surrounding_pheromones = dict()
        self.last_move = None
        self.picked_item = None
        self.pheromone = None
        self.forced_next = None

    def __repr__(self):
        return f'Worker ant, id = {self.id}, located in {self.position}'

    def set_position(self, pos: tuple, side_effect: bool = False):
        if side_effect:
            self.path.append(self.position)
            self.cycles_from += 1
            self.cycles_from = np.append(self.cycles_from, 0)
        self.position = pos

    def update_position(self, move: str, removing: bool = False):

        moved = None

        if not removing:
            self.path.append(self.position)
            if len(self.path) > 1:
                self.remove_cycles_from_path()

        if move == 'increase_x':
            self.position = (self.position[0] + 1, self.position[1], self.position[2])
        if move == 'decrease_x':
            self.position = (self.position[0] - 1, self.position[1], self.position[2])

        if move == 'increase_z':
            self.position = (self.position[0], self.position[1], self.position[2] + 1)
        if move == 'decrease_z':
            self.position = (self.position[0], self.position[1], self.position[2] - 1)

        if move == 'climb_up':
            self.position = (self.position[0], self.position[1] - 1, self.position[2])
        if move == 'climb_down':
            self.position = (self.position[0], self.position[1] + 1, self.position[2])

        if move == 'pick_item':
            moved = self.pick_item()

        if move == 'release_item':
            moved = self.release_item()

        self.cycles_from += 1
        self.cycles_from = np.append(self.cycles_from, 0)
        self.last_move = move

        return moved

    def release_item(self):

        assert self.picked_item == 2
        moved = None

        if Ant._world_type == '2d':
            self.map[self.position[1], self.position[0]] = self.picked_item
        else:
            self.map[self.position[2], self.position[1], self.position[0]] = self.picked_item

        if self.position[1] > 0:
            moved = True
            self.position = (self.position[0], self.position[1] - 1, self.position[2])

        self.picked_item = None
        self.pheromone = None

        return moved

    def pick_item(self):
        item = self.surroundings['current']
        assert item == 2

        moved = None
        if Ant._world_type == '2d':
            self.map[self.position[1], self.position[0]] = 1
        else:
            self.map[self.position[2], self.position[1], self.position[0]] = 1

        if self.position[1] < self.map.shape[0] - 1:
            moved = True
            self.position = (self.position[0], self.position[1] + 1, self.position[2])

        self.picked_item = item
        self.pheromone = 'resource'

        return moved

    def future_position(self, move):
        if move == 'increase_x':
            return self.position[0] + 1, self.position[1], self.position[2]
        if move == 'decrease_x':
            return self.position[0] - 1, self.position[1], self.position[2]

        if move == 'increase_z':
            return self.position[0], self.position[1], self.position[2] + 1
        if move == 'decrease_z':
            return self.position[0], self.position[1], self.position[2] - 1

        if move == 'climb_up':
            return self.position[0], self.position[1] - 1, self.position[2]
        if move == 'climb_down':
            return self.position[0], self.position[1] + 1, self.position[2]

    def remove_cycles_from_path(self):

        i = -1
        last_position = self.path[-1]
        for idx in range(len(self.path) - 1):
            if self.path[idx] == last_position:
                i = idx
                break

        if i != -1:
            self.path = self.path[:i + 1]
            self.cycles_from = self.cycles_from[:i + 1]
            self.cycles_from[-1] = 0

    def set_surrounding_pheromones(self, phero):
        self.surrounding_pheromones = phero

    def find_personal_possibilities(self):
        personal_moves = copy.deepcopy(Ant._possible_actions)
        clause = False

        # Disabling climbing up and down if the surface mode is enabled
        if Ant._resource_spawn == 'surface':
            personal_moves.remove('climb_up')
            personal_moves.remove('climb_down')

        # Removing moves that will cause the ant to go out of bounds (until line 300)
        # Removing all moves that could result in the ant going out of the map
        if self.position[0] == 0:
            personal_moves.remove('decrease_x')
        if self.position[0] == self.map.shape[0] - 1:
            personal_moves.remove('increase_x')

        if self.position[2] == 0 and 'decrease_z' in personal_moves:
            personal_moves.remove('decrease_z')
        if self.position[2] == self.map.shape[0] - 1 and 'increase_z' in personal_moves:
            personal_moves.remove('increase_z')

        # The ant can pick up only resource item
        if self.surroundings['current'] != 2 and 'pick_item' in personal_moves:
            personal_moves.remove('pick_item')

        # An ant can climb up or down (movement along y-axis) if this does not mean it exits from the map
        # and it does not go to a voxel in which there is no ground
        if self.position[1] == self.map.shape[0] - 1 and 'climb_down' in personal_moves:
            personal_moves.remove('climb_down')
        if (('up' in self.surroundings and self.surroundings['up'] == 0 and 'climb_up' in personal_moves)
                or 'up' not in self.surroundings):
            personal_moves.remove('climb_up')

        # Removing moves that would cause the ant to float in the air (go to 0 voxels)
        # Ants cannot move to a 0 spot (it would mean they can float in the air)
        if ('left' in self.surroundings and self.surroundings['left'] == 0 and 'decrease_x'
                in personal_moves):
            personal_moves.remove('decrease_x')
        if ('right' in self.surroundings and self.surroundings['right'] == 0 and 'increase_x'
                in personal_moves):
            personal_moves.remove('increase_x')

        if ('forward' in self.surroundings and self.surroundings['forward'] == 0 and 'decrease_z'
                in personal_moves and 'decrease_z' in personal_moves):
            personal_moves.remove('decrease_z')
        if ('backward' in self.surroundings and self.surroundings['backward'] == 0 and 'decrease_z'
                in personal_moves and 'increase_z' in personal_moves):
            personal_moves.remove('increase_z')

        # Ant cannot pick items at their spawn position (it's where they leave resources),
        # cannot pick items when they already hold some items and cannot pick itemes other than resources
        if ((self.position == self.spawn_position) or (self.picked_item is not None) or
                (self.surroundings['current'] != 2)) and 'pick_item' in personal_moves:
            personal_moves.remove('pick_item')

        # Ants cannot release items if they do not hold something
        if self.picked_item is None and 'release_item' in personal_moves:
            personal_moves.remove('release_item')

        if (Ant._resource_spawn == 'surface' and 'climb_down' not in personal_moves and
                (('left' in self.surroundings and self.surroundings['left'] == 0) or
                 ('right' in self.surroundings and self.surroundings['right'] == 0) or
                 ('forward' in self.surroundings and self.surroundings['forward'] == 0) or
                 ('backward' in self.surroundings and self.surroundings['backward'] == 0))):
            clause = True
            personal_moves.append('climb_down')
            keys = ['left', 'right', 'forward', 'backward']
            keys = [item for item in keys if (item in self.surroundings and self.surroundings[item] == 0)]
            inner_probs = dict()

            for key in keys:
                inner_probs[key] = 1 / len(keys)

            l = list(inner_probs.keys())
            if len(l) == 1:
                self.forced_next = surrounding_to_move[l[0]]
            else:
                cum_sum = 0
                p = random.random()
                for key in inner_probs:
                    cum_sum += inner_probs[key]
                    if p < cum_sum:
                        self.forced_next = surrounding_to_move[key]
                        break

        return personal_moves, clause

    def move(self):

        move = ''

        if self.picked_item is None:

            if self.forced_next is not None:
                move = self.forced_next
                self.forced_next = None
                moved = self.update_position(move)
                return move, moved

            # forcing ant to climb to surface if the ant is underground when the surface mode is enabled
            if (Ant._resource_spawn == 'surface' and 'up' in self.surroundings and
                    self.surroundings['up'] in [1, 2] and self.last_move != 'climb_down'):
                moved = self.update_position('climb_up')
                return move, moved

            if self.position != self.spawn_position and self.surroundings['current'] == 2:
                moved = self.update_position('pick_item')
                return 'pick_item', moved

            else:
                personal_possible_moves, clause = self.find_personal_possibilities()
                probs = dict()

                for item in personal_possible_moves:
                    probs[item] = 1 / len(personal_possible_moves)

                assert len(personal_possible_moves) > 0

                if len(list(probs.keys())) == 1:
                    move = list(probs.keys())[0]
                    if move != 'climb_down' and clause:
                        self.forced_next = None

                else:
                    # make sure the Ant does not return to where it previously was
                    if self.last_move is not None:
                        probs = lower_probs(probs, [invert(self.last_move)])

                    moves_to_increase = list()

                    for key in self.surrounding_pheromones:
                        moves_to_increase.append(surrounding_to_move[key])

                    if 0 < len(moves_to_increase) < len(list(probs.keys())):
                        probs = increase_probs_with_pheromones(probs, moves_to_increase)

                    moves_to_increase = list()

                    for key in self.surroundings:
                        if (key != 'current' and key != 'up-left' and key != 'up-right' and
                                self.surroundings[key] == 2 and
                                self.future_position(surrounding_to_move[key]) != self.spawn_position):
                            moves_to_increase.append(surrounding_to_move[key])

                    # increase the probability to go to a place where resources are present
                    if 0 < len(moves_to_increase) < len(list(probs.keys())):
                        probs = increase_probs(probs, moves_to_increase)

                    choice = random.random()
                    cum_sum = 0

                    for key in probs:
                        cum_sum += probs[key]
                        if choice < cum_sum:
                            move = key
                            break

                    if move != 'climb_down' and clause:
                        self.forced_next = None

            moved = self.update_position(move)

        else:
            if self.picked_item != 2:
                print('Something went wrong during item picking')
                exit(10)

            if self.position == self.spawn_position:
                moved = self.update_position('release_item')
                self.path = list()
                self.path.append(self.position)
                self.cycles_from = np.array(list())
                self.cycles_from = np.append(self.cycles_from, 0)
                # to force the ant back to the spawn position after it releases the item
                # self.forced_next = 'climb_down'
                return 'release_item', moved

            else:
                if len(self.path) > 0:
                    move = find_move(self.position, self.path[-1])
                    if move == 0:
                        self.path = self.path[:-1]
                        self.cycles_from = self.cycles_from[:-1]
                        move = find_move(self.position, self.path[-1])
                    self.path = self.path[:-1]
                    self.cycles_from = self.cycles_from[:-1]
                    self.update_position(move, removing=True)
                    return move, None

                else:
                    move = find_move(self.position, self.spawn_position)
                    if move != 0:
                        moved = self.update_position(move)
                        return move, moved
                    print(f'Error, path length < 1')
                    exit(42)

        return move, moved

    def reset_path_based_on_spawn(self):
        if self.position == self.spawn_position:
            self.path = list()
            self.path.append(self.spawn_position)
            self.cycles_from = np.array(list())
            self.cycles_from = np.append(self.cycles_from, 0)

    def get_spawn_position(self):
        return self.spawn_position

    def get_id(self):
        return self.id

    def release_pheromones(self):
        return self.pheromone

    def get_knowledge_from_queen(self,
                                 new_map: np.ndarray):
        self.map = new_map

    def get_spawn_position(self):
        return self.spawn_position

    def pass_knowledge_to_queen(self):
        cycle_map = np.ones_like(self.map)

        for i in range(len(self.path)):
            if Ant._world_type == '2d':
                cycle_map[self.path[i][1], self.path[i][0]] = self.cycles_from[i]
            else:
                cycle_map[self.path[i][2], self.path[i][1], self.path[i][0]] = self.cycles_from[i]

        return self.map, cycle_map

    def get_picked_item(self):
        return self.picked_item
