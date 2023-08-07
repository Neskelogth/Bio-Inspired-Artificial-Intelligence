import abc

import numpy as np


class Ant:
    possible_moves = ['increase_x', 'decrease_x', 'increase_z', 'decrease_z', 'dig_up', 'dig_down']
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
        _world_type = world_type
        _resource_spawn = resource_spawn

    def __init__(self,
                 **kwagrs):
        """

        :param kwagrs: dictionary with all the necessary information to build an ant.
                       Required keys are: type (string), size (int) and position (tuple of ints)

        """

        assert 'type' in kwagrs and 'size' in kwagrs and 'position' in kwagrs, ('The required keys for the ant construction are \'type\', \'position\', and \'size\'. At least one of them is missing, please provide it when constructing the ant object.')
        assert type(kwagrs['type']) == str
        assert type(kwagrs['size']) == int
        assert (type(kwagrs['position']) == tuple and len(kwagrs['position']) == 3 and
                type(kwagrs['position'][0]) == type(kwagrs['position'][1]) == type(kwagrs['position'][2]) == int), ('The position parameter should be a tuple of three integers')

        self.type = kwagrs['type']
        self.size = kwagrs['size']
        self.position = kwagrs['position']
        # A value of 0 represents land, 1 represents air, 2 represents a resource, -1 represents unknown
        self.map = np.zeros((self.size, self.size, self.size)) - 1
        self.path_from_beginning = list()

    def get_position(self):
        return self.position

    @abc.abstractmethod
    def move(self):
        raise NotImplementedError

    @abc.abstractmethod
    def release_pheromones(self):
        raise NotImplementedError
