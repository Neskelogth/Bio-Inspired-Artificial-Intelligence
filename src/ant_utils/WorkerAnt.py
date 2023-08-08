from .Ant import Ant
import numpy as np


class WorkerAnt(Ant):
    def __init__(self, **kwargs):
        super().__init__(type='worker', **kwargs)
        self.cycles_from = np.zeros_like(self.map, dtype=np.uint8)
        self.path = list()
        self.path.append(self.position)

    def __repr__(self):
        return f'Worker ant, located in {self.position}'

    def move(self):
        print('TODO')

    def release_pheromones(self):
        print('TODO')

    def pick_item_up(self):
        print('TODO')

    def release_item(self):
        print('TODO')

    def get_knowledge_from_queen(self, map):
        self.map = map

    def pass_knowledge_to_queen(self):
        return self.map
