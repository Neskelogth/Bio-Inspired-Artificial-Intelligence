from .Ant import Ant
import numpy as np
from more_itertools import locate


class QueenAnt(Ant):

    def __init__(self,
                 **kwagrs):
        """
        :param kwagrs: dictionary with all the necessary information to build a queen ant.
                       Required keys are: size (int) and position (tuple of ints)
        """
        super().__init__(type='queen', **kwagrs)
        self.passed_time = np.zeros_like(self.map, dtype=np.uint8)
        self.max_iter = kwagrs['max_iter']
        self.blocks = None

    def __repr__(self):
        return f'Queen ant, located in {self.position}'

    def fully_explored(self, spawns, counter):

        if counter > self.max_iter:
            return True, 'max_iter'

        surface_explored = True
        underground_explored = True
        all_spawns = True
        block_counter = 0
        surface = np.array(list(), dtype=np.int8)
        res_blocks = list()

        if Ant._world_type == '2d':
            for i in range(len(self.map)):
                for j in range(len(self.map[i])):
                    if self.map[i][j] == 2:
                        block_counter += 1
                        res_blocks.append((j, i, 0))

            all_spawn_blocks = [res_block in spawns for res_block in res_blocks]
            if not all(all_spawn_blocks):
                all_spawns = False

            for i in range(len(self.map)):
                col = self.map[:, i]
                zeros_positions = list(locate(col, lambda x: x == 0))
                index_to_pop = list()
                for j in range(len(zeros_positions) - 1):
                    if zeros_positions[j + 1] == zeros_positions[j] + 1:
                        index_to_pop.append(j)
                for idx in range(len(index_to_pop) - 1, -1, -1):
                    zeros_positions.pop(index_to_pop[idx])

                if len(zeros_positions) == 0:
                    surface_explored = False
                    break
                assert len(zeros_positions) == 1, 'Something went wrong in the queen map update'

                idx = zeros_positions[0]
                if idx + 1 < len(self.map) and col[idx + 1] != 1 and col[idx + 1] != 2:
                    surface_explored = False
                    break
                surface = np.append(surface, idx)

            if Ant._resource_spawn == 'normal':
                try:
                    surface = surface.reshape((self.map.shape[1:]))
                except:
                    underground_explored = False

                if underground_explored:
                    for i in range(len(self.map)):
                        broken = False
                        ith_col = self.map[:, i]
                        for j in range(surface[i], len(self.map)):
                            if ith_col[j] == -1:
                                surface_explored = False
                                broken = True
                                break
                        if broken:
                            break
        else:
            for i in range(len(self.map)):
                for j in range(len(self.map[i])):
                    for k in range(len(self.map[i][j])):
                        if self.map[i][j][k] == 2:
                            block_counter += 1
                            res_blocks.append((k, j, i))

            all_spawn_blocks = [res_block in spawns for res_block in res_blocks]

            if not all(all_spawn_blocks):
                all_spawns = False

            for i in range(len(self.map)):
                broken = False
                line_map = np.array(list())
                for j in range(len(self.map[i])):
                    jth_col = self.map[i, :, j]
                    zeros_positions = list(locate(jth_col, lambda x: x == 0))
                    index_to_pop = list()
                    for k in range(len(zeros_positions) - 1):
                        if zeros_positions[k + 1] == zeros_positions[k] + 1:
                            index_to_pop.append(k)
                    for idx in range(len(index_to_pop) - 1, -1, -1):
                        zeros_positions.pop(index_to_pop[idx])

                    if len(zeros_positions) == 0:
                        return False, 'criterion'

                    assert len(zeros_positions) == 1, f'Something went wrong in the queen map update {self.map}'

                    idx = zeros_positions[0]
                    if self.map[i][idx][j] + 1 not in [1, 2]:
                        surface_explored = False
                        broken = True
                        break

                    line_map = np.append(line_map, idx)
                surface = np.append(surface, line_map)

                if broken:
                    break

            if Ant._resource_spawn == 'normal':
                try:
                    surface = surface.reshape((self.map.shape[1:])).astype(np.int8)
                except:
                    underground_explored = False

                for i in range(len(self.map)):
                    broken = False
                    for j in range(len(self.map[i])):
                        for k in range(surface[i, j], len(self.map[i, j])):
                            if self.map[i, k, j] == -1:
                                underground_explored = False
                                broken = True
                                break
                        if broken:
                            break
                    if broken:
                        break

        if Ant._resource_spawn == 'surface':
            return block_counter <= len(spawns) and all_spawns and surface_explored, 'criterion'
        else:
            return (block_counter <= len(spawns) and all_spawns and
                    surface_explored and underground_explored, 'criterion')

    def update_passed_time(self):
        self.passed_time += 1

    def move(self):
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
                        # if the queen already knows the state of that voxel.
                        # The voxel gets effectively updated only if the time passed from the worker's
                        # measure is lesser than the one from the queen's previous measure
                        else:
                            if cycles_from[i][j] <= self.passed_time[i][j]:
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
                            else:
                                if cycles_from[i][j][k] < self.passed_time[i][j][k]:
                                    self.map[i][j][k] = worker_map[i][j][k]
                                    self.passed_time[i][j][k] = cycles_from[i][j][k]

    def get_map(self):
        return self.map
