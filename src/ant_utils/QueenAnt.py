from .Ant import Ant


class QueenAnt(Ant):

    def __init__(self,
                 **kwagrs):
        """
        :param kwagrs: dictionary with all the necessary information to build a queen ant.
                       Required keys are: size (int) and position (tuple of ints)
        """
        super().__init__(type='queen', **kwagrs)

    def __repr__(self):
        return f'Queen ant, located in {self.position}'

    def move(self):
        return None

    def release_pheromones(self):
        return None

    def get_knowledge(self, map):
        # TODO
        pass

    def pass_knowledge(self):
        return self.map
