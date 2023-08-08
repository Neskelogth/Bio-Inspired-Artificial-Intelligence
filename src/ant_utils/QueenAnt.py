from .Ant import Ant
import numpy as np


class QueenAnt(Ant):

    def __init__(self,
                 **kwagrs):
        """
        :param kwagrs: dictionary with all the necessary information to build a queen ant.
                       Required keys are: size (int) and position (tuple of ints)
        """
        super().__init__(type='queen', **kwagrs)
        self.nest_position = None
        self.passed_time = np.zeros_like(self.map, dtype=np.uint8)

    def __repr__(self):
        return f'Queen ant, located in {self.position}, '

    def update_passed_time(self):
        self.passed_time += 1

    def move(self):
        if self.nest_position is not None:
            #TODO
            pass

        return None

    def release_pheromones(self):
        return None

    def get_knowledge_from_worker(self,
                                  worker_map: np.ndarray,
                                  cycles_from: np.ndarray):
        # Seems to be working
        if Ant._world_type == '2d':
            for i in range(self.size):
                for j in range(self.size):
                    if worker_map[i][j] != -1:
                        # the queen still does not know the state of that voxel
                        if self.map[i][j] == -1:
                            self.map[i][j] = worker_map[i][j]
                            self.passed_time[i][j] = cycles_from[i][j]
                        # if the queen already knows the state of that voxel, if the voxel is 1 (ground) it may
                        # be updated. The voxel gets effectively updated only if the time passed from the worker's
                        # measure is lesser than the one from the previous measure
                        elif self.map[i][j] == 1 and cycles_from[i][j] < self.passed_time[i][j]:
                            self.map[i][j] = worker_map[i][j]
                            self.passed_time[i][j] = cycles_from[i][j]
        else:
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        if worker_map[i][j][k] != -1:
                            # the queen still does not know the state of that voxel
                            if self.map[i][j][k] == -1:
                                self.map[i][j][k] = worker_map[i][j][k]
                                self.passed_time[i][j][k] = cycles_from[i][j][k]
                            # if the queen already knows the state of that voxel, if the voxel is 1 (ground) it may
                            # be updated. The voxel gets effectively updated only if the time passed from the worker's
                            # measure is lesser than the one from the previous measure
                            elif self.map[i][j][k] == 1 and cycles_from[i][j][k] < self.passed_time[i][j][k]:
                                self.map[i][j][k] = worker_map[i][j][k]
                                self.passed_time[i][j][k] = cycles_from[i][j][k]

    def get_map(self):
        return self.map
