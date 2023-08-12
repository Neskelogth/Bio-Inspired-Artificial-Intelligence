import copy
import random

import numpy as np

from .Ant import Ant

surrounding_to_move = {
    'right': 'increase_x',
    'left': 'decrease_x',

    'forward': 'increase_z',
    'backward': 'decrease_z',

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
    to_redistribute = 0
    for move in moves:
        if move in probs:
            to_redistribute += (probs[move] - final_value)

    keys_to_increase = len(list(probs.keys())) - len(moves)
    prob_increase = to_redistribute / keys_to_increase
    for key in list(probs.keys()):
        if key not in moves:
            probs[key] += prob_increase
        else:
            probs[key] = final_value

    return probs


def increase_probs(probs: dict, moves: list, final_value: float = 0.9):
    diff = final_value
    for key in probs:
        if key in moves:
            diff -= probs[key]

    value_to_add = diff / len(moves)
    value_to_remove = diff / (len(list(probs.keys())) - len(moves))

    for key in probs:
        if key in moves:
            probs[key] = probs[key] + value_to_add
        else:
            probs[key] = probs[key] - value_to_remove

    return probs


def increase_probs_with_pheromones(probs):
    # TODO
    return probs


def find_move(position_1: tuple, position_2: tuple):

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

    print('Something went wrong in move computation to go back to the nest')
    exit(42)


class WorkerAnt(Ant):
    def __init__(self, **kwargs):
        super().__init__(type='worker', **kwargs)
        self.spawn_position = self.position
        self.id = kwargs['id']
        self.cycles_from = np.array(list(), dtype=np.uint8)
        self.path = list()

        self.surrounding_pheromones = None
        self.last_move = None
        self.picked_item = None
        self.pheromone = None

    def __repr__(self):
        return f'Worker ant, id = {self.id}, located in {self.position}'

    def is_suitable_location(self):

        if 'up' in self.surroundings and self.surroundings['up'] == 0:
            return True
        return False

    def update_position(self, move: str):

        self.cycles_from += 1
        self.cycles_from = np.append(self.cycles_from, 0)
        self.path.append(self.position)
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

        if move == 'pick_up_item':
            self.pick_item(self.surroundings['current'])

        if move == 'release_item':
            self.release_item()

    def release_item(self):

        print('releasing')
        if Ant._world_type == '2d':
            self.map[self.position[1], self.position[0]] = self.picked_item
        else:
            self.map[self.position[2], self.position[1], self.position[0]] = self.picked_item

        if self.position[1] > 0:
            self.position = (self.position[0], self.position[1] - 1, self.position[2])

        self.picked_item = None
        self.pheromone = None

    def pick_item(self, item):
        print('picking')
        if Ant._world_type == '2d':
            self.map[self.position[1], self.position[0]] = 0
        else:
            self.map[self.position[2], self.position[1], self.position[0]] = 0

        if self.position[1] < self.map.shape[0]:
            self.position = (self.position[0], self.position[1] + 1, self.position[2])

        self.picked_item = item
        if item == 2:
            self.pheromone = 'resource'

    def remove_cycles_from_path(self):

        last_position = self.path[-1]
        for idx in range(len(self.path) - 1):
            if self.path[idx] == last_position:
                self.path = self.path[:idx + 1]
                self.cycles_from = self.cycles_from[:idx + 1]
                self.cycles_from[-1] = 0

    def move(self):

        move = ''
        if self.picked_item is not None:

            if self.picked_item != 1 and self.picked_item != 2:
                print('Something went wrong during item picking')
                exit(10)

            if ((self.picked_item == 2 and self.position == self.spawn_position) or
                    (self.picked_item == 1 and self.is_suitable_location())):
                self.update_position('release_item')
                self.path = np.append(self.path, self.position)
                return 'release_item'

            else:
                if len(self.path) > 0:
                    move = find_move(self.position, self.path[-1])
                    self.path = self.path[:-1]
                    self.last_move = move
                    self.update_position(move)
                    return move
                else:
                    self.update_position('release_item')
                    self.path = np.append(self.path, self.position)
                    return 'release_item'

        else:
            # If the ant is on a resource, it picks it up
            if self.surroundings['current'] == 2:
                self.update_position('pick_item')
                return 'pick_item'

            personal_possible_moves = copy.deepcopy(Ant._possible_actions)
            # Removing all moves that could result in the ant going out of the map
            if self.position[0] == 0:
                personal_possible_moves.remove('decrease_x')
            if self.position[0] == self.map.shape[0] - 1:
                personal_possible_moves.remove('increase_x')

            if self.position[2] == 0 and 'decrease_z' in personal_possible_moves:
                personal_possible_moves.remove('decrease_z')
            if self.position[2] == self.map.shape[0] - 1 and 'increase_z' in personal_possible_moves:
                personal_possible_moves.remove('increase_z')

            # An item can be released only if it is being held,
            # so if there is no item held then it's not possible to release it
            # Similarly, if an item is held, it's not possible to pick up another
            if self.picked_item is None:
                personal_possible_moves.remove('release_item')
            else:
                personal_possible_moves.remove('pick_item')

            # If the surface mode is enabled, then the ant can pick up only resource item
            if (Ant._resource_spawn == 'surface' and self.surroundings['current'] != 2
                    and 'pick_item' in personal_possible_moves):
                personal_possible_moves.remove('pick_item')

            # An ant can climb up or down (movement along y-axis) if this does not mean it exits from the map.
            # An additional condition is set later
            if self.position[1] == self.map.shape[0] - 1 and 'climb_down' in personal_possible_moves:
                personal_possible_moves.remove('climb_down')
            if (('up' in self.surroundings and self.surroundings['up'] == 0 and 'climb_up' in personal_possible_moves)
                    or 'up' not in self.surroundings):
                personal_possible_moves.remove('climb_up')

            # Ants cannot move to a 0 spot (it would mean they can float in the air)
            if ('left' in self.surroundings and self.surroundings['left'] == 0 and 'decrease_x'
                    in personal_possible_moves):
                personal_possible_moves.remove('decrease_x')
            if ('right' in self.surroundings and self.surroundings['right'] == 0 and 'increase_x'
                    in personal_possible_moves):
                personal_possible_moves.remove('increase_x')

            if ('forward' in self.surroundings and self.surroundings['forward'] == 0 and 'decrease_z'
                    in personal_possible_moves and 'decrease_z' in personal_possible_moves):
                personal_possible_moves.remove('decrease_z')
            if ('backward' in self.surroundings and self.surroundings['backward'] == 0 and 'decrease_z'
                    in personal_possible_moves and 'increase_z' in personal_possible_moves):
                personal_possible_moves.remove('increase_z')

            # In surface mode, if other moves are possible without resulting in the ant floating
            # do not consider climbing up and down
            if (Ant._resource_spawn == 'surface' and 'climb_up' in personal_possible_moves and
                    (('increase_x' in personal_possible_moves and 'decrease_x' in personal_possible_moves) or
                     ('increase_z' in personal_possible_moves and 'decrease_z' in personal_possible_moves) or
                     ('increase_x' in personal_possible_moves and 'left' not in self.surroundings) or
                     ('decrease_x' in personal_possible_moves and 'right' not in self.surroundings) or
                     ('increase_z' in personal_possible_moves and 'backward' not in self.surroundings) or
                     ('decrease_z' in personal_possible_moves and 'forward' not in self.surroundings))):
                personal_possible_moves.remove('climb_up')

            if (Ant._resource_spawn == 'surface' and 'climb_down' in personal_possible_moves and
                    (('increase_x' in personal_possible_moves and 'decrease_x' in personal_possible_moves) or
                     ('increase_z' in personal_possible_moves and 'decrease_z' in personal_possible_moves) or
                     ('increase_x' in personal_possible_moves and 'left' not in self.surroundings) or
                     ('decrease_x' in personal_possible_moves and 'right' not in self.surroundings) or
                     ('increase_z' in personal_possible_moves and 'backward' not in self.surroundings) or
                     ('decrease_z' in personal_possible_moves and 'forward' not in self.surroundings))):
                personal_possible_moves.remove('climb_down')

            if 'up' in self.surroundings and self.surroundings['up'] == 2 and 'climb_up' not in personal_possible_moves:
                personal_possible_moves.append('climb_up')

            if ('down' in self.surroundings and self.surroundings['down'] == 2
                    and 'climb_down' not in personal_possible_moves):
                personal_possible_moves.append('climb_up')

            probs = dict()
            for item in personal_possible_moves:
                probs[item] = 1 / len(personal_possible_moves)

            if len(list(probs.keys())) == 1:
                move = list(probs.keys())[0]

            else:
                # make sure the Ant does not return to where it previously was
                if self.last_move is not None:
                    probs = lower_probs(probs, [invert(self.last_move)])

                moves_to_increase = list()

                for key in self.surroundings:
                    if self.surroundings[key] == 2:
                        moves_to_increase.append(surrounding_to_move[key])

                # TODO increase with pheromones

                # increase the probability to go to a place where resources are present
                if len(moves_to_increase) > 0:
                    probs = increase_probs(probs, moves_to_increase)

                keys = list(probs.keys())
                keys.sort()
                probs = {i: probs[i] for i in keys}
                choice = random.random()
                cum_sum = 0

                for key in probs:
                    cum_sum += probs[key]
                    if choice < cum_sum:
                        move = key
                        break

                self.last_move = move

            self.update_position(move)
        return move

    def get_id(self):
        return self.id

    def release_pheromones(self):
        return self.pheromone

    def get_knowledge_from_queen(self,
                                 new_map: np.ndarray):
        self.map = new_map

    def pass_knowledge_to_queen(self):
        cycle_map = np.ones_like(self.map)

        for i in range(len(self.path)):
            if Ant._world_type == '2d':
                cycle_map[self.path[i][1], self.path[i][0]] = self.cycles_from[i]
            else:
                cycle_map[self.path[i][2], self.path[i][1], self.path[i][0]] = self.cycles_from[i]

        return self.map, cycle_map
