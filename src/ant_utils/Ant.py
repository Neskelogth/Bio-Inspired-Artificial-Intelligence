import abc
import numpy as np


class Ant(abc.ABC):
    _possible_actions = ['increase_x', 'decrease_x', 'increase_z', 'decrease_z',
                         'climb_up', 'climb_down',  # climb up and down are for movement on a vertical wall
                         'pick_item', 'release_item']
    _world_type = ''
    _resource_spawn = ''

    @classmethod
    def set_hyperparameters(cls,
                            world_type: str = '2d',
                            resource_spawn: str = 'surface'):
        """

        :type world_type: string
        :param world_type: Type of world used in the current configuration. Should be either '2d' or '3d'
        :type resource_spawn: string
        :param resource_spawn: Type of spawn method for resources. Should be either 'normal' or 'surface'

            sets the type of world and the resource spawn method of this world
        """
        world_type = world_type.lower()
        resource_spawn = resource_spawn.lower()
        assert world_type in ['2d', '3d'], ('The world type should be either \'2d\' or \'3d\'. '
                                            'Please check your code to see if you pass the parameter correctly')
        assert resource_spawn in ['normal', 'surface'], ('The world type should be either \'surface\' or \'normal\'. '
                                                         'Please check your code to see if you pass the '
                                                         'parameter correctly')
        cls._world_type = world_type
        cls._resource_spawn = resource_spawn
        if world_type == '2d':
            cls._possible_actions.remove('increase_z')
            cls._possible_actions.remove('decrease_z')

    def __init__(self,
                 **kwagrs):
        """

        :param kwagrs: dictionary with all the necessary information to build an ant.
                       Required keys are: type (string), size (int) and position (tuple of ints)

        """

        assert 'type' in kwagrs and 'size' in kwagrs and 'position' in kwagrs, (
            'The required keys for the ant construction are \'type\', \'position\', and \'size\'. '
            'At least one of them is missing, please provide it when constructing the ant object.')
        assert isinstance(kwagrs['type'], str), 'The type argument must be a string'
        assert isinstance(kwagrs['size'], int), 'The size argument must be an integer'
        assert (isinstance(kwagrs['position'], tuple) and len(kwagrs['position']) == 3 and
                isinstance(kwagrs['position'][0], int) and isinstance(kwagrs['position'][1], int) and isinstance(
                    kwagrs['position'][2], int)), 'The position parameter should be a tuple of three integers'

        self.type = kwagrs['type']
        self.size = kwagrs['size']
        self.position = kwagrs['position']
        # A value of 0 represents land, 1 represents air, 2 represents a resource, -1 represents unknown
        if Ant._world_type == '2d':
            self.map = np.zeros((self.size, self.size), dtype=np.int8) - 1
        else:
            self.map = np.zeros((self.size, self.size, self.size), dtype=np.int8) - 1
        self.surroundings = None

    def set_position(self, pos: tuple):
        self.position = pos

    def set_surroundings(self, surr: dict):
        self.surroundings = surr

        if Ant._world_type == '2d':
            self.map[self.position[1], self.position[0]] = surr['current']
            if 'left' in surr:
                self.map[self.position[1], self.position[0] - 1] = surr['left']
            if 'right' in surr:
                self.map[self.position[1], self.position[0] + 1] = surr['right']

            if 'up' in surr:
                self.map[self.position[1] - 1, self.position[0]] = surr['up']
            if 'down' in surr:
                self.map[self.position[1] + 1, self.position[0]] = surr['down']

        else:
            self.map[self.position[2], self.position[1], self.position[0]] = surr['current']
            if 'left' in surr:
                self.map[self.position[2], self.position[1], self.position[0] - 1] = surr['left']
            if 'right' in surr:
                self.map[self.position[2], self.position[1], self.position[0] + 1] = surr['right']

            if 'up' in surr:
                self.map[self.position[2], self.position[1] - 1, self.position[0]] = surr['up']
            if 'down' in surr:
                self.map[self.position[2], self.position[1] + 1, self.position[0]] = surr['down']

            if 'forward' in surr:
                self.map[self.position[2] - 1, self.position[1], self.position[0]] = surr['forward']
            if 'backward' in surr:
                self.map[self.position[2] + 1, self.position[1], self.position[0]] = surr['backward']

    def get_position(self):
        return self.position

    @abc.abstractmethod
    def move(self):
        raise NotImplementedError('The method fell to the base class Ant, the subclass '
                                  'that called this method should implement it')

    @abc.abstractmethod
    def release_pheromones(self):
        raise NotImplementedError('The method fell to the base class Ant, the subclass '
                                  'that called this method should implement it')
