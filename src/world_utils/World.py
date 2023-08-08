from perlin_noise import PerlinNoise
import numpy as np
import math
import random
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib.pyplot as plt


class World:
    def __init__(self, kwargs):

        self.dim = kwargs['size']
        self.multiplier = kwargs['multiplier']
        self.offset = kwargs['offset']
        self.octaves = kwargs['octaves']
        self.seed = kwargs['seed']
        self._2d = kwargs['2d']
        self.mode = kwargs['resource_placement']
        self.max_tries = kwargs['max_tries']
        self.heap_dimension_x = kwargs['heap_dimension_x']
        self.heap_dimension_y = kwargs['heap_dimension_y']
        self.heap_dimension_z = kwargs['heap_dimension_z']
        self.cutoff_distance = kwargs['cutoff_distance']
        self.epsilon = kwargs['epsilon']
        self.map = None
        self.surface = None

        self.generate_surface()
        self.generate_world()
        self.place_resources()

    def __repr__(self):
        if self._2d:
            plt.imshow(self.map)
            plt.show()
            return ''
        else:
            raise NotImplementedError

    def get_surface_shape(self):
        return self.map[0].shape

    def generate_surface(self):

        noise = PerlinNoise(octaves=self.octaves, seed=self.seed)
        x_pixel, y_pixel = self.dim, self.dim

        self.surface = list()

        if self._2d:
            for i in tqdm(range(x_pixel), desc='Generating surface level'):
                self.surface.append(math.ceil(noise(i / x_pixel) * self.multiplier) + self.offset)

        else:
            for j in tqdm(range(x_pixel), desc='Generating surface level'):
                temp = list()
                for i in range(y_pixel):
                    temp.append(math.ceil(noise([i / x_pixel, j / y_pixel]) * self.multiplier) + self.offset)
                self.surface.append(temp)

    def generate_world(self):

        if self._2d:
            self.map = np.zeros((self.dim, self.dim), dtype=np.uint8)
            for i in range(len(self.surface)):
                self.map[i][:self.surface[i]] = np.array([1] * self.surface[i])

        else:
            self.map = np.zeros((self.dim, self.dim, self.dim), dtype=np.uint8)
            for i in range(self.dim):
                for j in range(self.dim):
                    self.map[i][j][:self.surface[i][j]] = np.array([1] * self.surface[i][j])

    def check_distances(self,
                        positions: np.ndarray,
                        candidate: tuple | int):

        x, y, z = None, None, None
        if type(candidate) == tuple:
            x = candidate[0]
            y = candidate[1]
            if len(candidate) == 3:
                z = candidate[2]
        else:
            x = candidate

        point = None
        candidate_point = None

        # surface mode and 2d world
        if len(positions.shape) == 1:
            for item in positions:
                if item == 0:
                    continue

                point = (item, 0)
                candidate_point = (x, 0)

                if (candidate_point is not None and point is not None and
                        distance.euclidean(point, candidate_point) < self.cutoff_distance):
                    return False

        # either surface mode for a 3d world or normal mode for 2d world
        elif positions.shape[1] == 2:

            for item in positions:
                if self.mode == 'normal' and candidate[1] >= sum(self.map[x]):
                    return False

                if all(item == 0):
                    continue

                point = (item[0], item[1])
                candidate_point = (x, y)

                if (candidate_point is not None and point is not None and
                        distance.euclidean(point, candidate_point) < self.cutoff_distance):
                    return False

        # normal mode for 3d world
        else:
            for item in positions:
                if self.map is not None and candidate[2] >= sum(self.map[x][y]):
                    return False

                if all(item == 0):
                    continue

                point = (item[0], item[1], item[2])
                candidate_point = (x, y, z)

                if (candidate_point is not None and point is not None and
                        distance.euclidean(point, candidate_point) < self.cutoff_distance):
                    return False

        return True

    def place_resources(self):

        if len(self.map.shape) == 2:

            # function to have 2 heaps at size 50, 50 at size 2500 and 100 at size 5000, polynomial interpolation
            heap_number = int((self.dim ** 2 / 12127500) + 1567 * self.dim / 80850 + 5000 / 4851)
            if self.mode == 'normal':
                heap_number *= 5

            if self.mode == 'surface':
                heap_spawn = np.zeros((heap_number,))
                for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                    counter = 0
                    candidate = int(random.random() * self.dim)
                    while not self.check_distances(heap_spawn, candidate):
                        if counter == self.max_tries:
                            break
                        counter += 1
                        candidate = int(random.random() * self.dim)
                    heap_spawn[i] = candidate

                for i in range(heap_number):
                    r = np.arange(start=math.ceil(heap_spawn[i] - self.heap_dimension_x / 2),
                                  stop=int(heap_spawn[i] + self.heap_dimension_x / 2) + 1, step=1)

                    r = [item for item in r if 0 <= item < self.dim]

                    for j in r:
                        surface_line_position = sum(self.map[j])
                        self.map[j][surface_line_position - 1] = 2

            else:
                heap_spawn = np.zeros((heap_number, 2))
                for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                    counter = 0
                    over_tries = False
                    candidate = (int(random.random() * self.dim), int(random.random() * self.dim))
                    while not self.check_distances(heap_spawn, candidate):
                        if counter == self.max_tries:
                            over_tries = True
                            break
                        counter += 1
                        candidate = (int(random.random() * self.dim), int(random.random() * self.dim))

                    heap_spawn[i][0], heap_spawn[i][1] = candidate[0], candidate[1]
                    if over_tries and candidate[1] >= sum(self.map[candidate[0]]):
                        heap_spawn[i][1] = sum(self.map[candidate[0]]) - 1

                for i in tqdm(range(heap_number)):
                    r1 = np.arange(start=math.ceil(heap_spawn[i][0] - self.heap_dimension_x / 2),
                                   stop=int(heap_spawn[i][0] + self.heap_dimension_x / 2) + 1, step=1)
                    r2 = np.arange(start=math.ceil(heap_spawn[i][1] - self.heap_dimension_y / 2),
                                   stop=int(heap_spawn[i][1] + self.heap_dimension_y / 2) + 1, step=1)

                    r1 = [item for item in r1 if 0 <= item < self.dim]
                    r2 = [item for item in r2 if 0 <= item < self.dim]

                    indexes_to_remove = list()

                    for j in range(len(r2)):
                        for k in r1:
                            if r2[j] >= sum(self.map[k]):
                                indexes_to_remove.append(j)

                    indexes_to_remove = list(set(indexes_to_remove))
                    indexes_to_remove.reverse()

                    for item in indexes_to_remove:
                        r2.pop(item)

                    for j in r1:
                        for k in r2:
                            self.map[j][k] = 2

            self.map = np.rot90(self.map)

        else:

            # function to have 100 heaps at size 50, 5000 heaps at size 1000,
            # 10000 at size 2500 and 100000 at size 5000, polynomial interpolation
            heap_number = int((-500000000 / 549989 + (11099980 * self.dim) / 549989 - self.dim ** 2 / 13749725) / 4)
            if self.mode == 'normal':
                heap_number *= 5

            if self.mode == 'surface':
                heap_spawn = np.zeros((heap_number, 2))
                for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                    counter = 0
                    candidate = (int(random.random() * self.dim), int(random.random() * self.dim))
                    while not self.check_distances(heap_spawn, candidate):
                        if counter == self.max_tries:
                            break
                        counter += 1
                        candidate = (int(random.random() * self.dim), int(random.random() * self.dim))
                    heap_spawn[i][0], heap_spawn[i][1] = candidate[0], candidate[1]

                for i in range(heap_number):
                    r1 = np.arange(start=math.ceil(heap_spawn[i][0] - self.heap_dimension_x / 2),
                                   stop=int(heap_spawn[i][0] + self.heap_dimension_x / 2) + 1, step=1)
                    r2 = np.arange(start=math.ceil(heap_spawn[i][1] - self.heap_dimension_y / 2),
                                   stop=int(heap_spawn[i][1] + self.heap_dimension_y / 2) + 1, step=1)

                    r1 = [item for item in r1 if 0 <= item < self.dim]
                    r2 = [item for item in r2 if (0 <= item < self.dim)]

                    for j in r1:
                        for k in r2:
                            self.map[j][k][sum(self.map[j][k]) - 1] = 2

            else:
                heap_spawn = np.zeros((heap_number, 3))
                for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                    counter = 0
                    over_tries = False
                    candidate = (int(random.random() * self.dim), int(random.random() * self.dim),
                                 int(random.random() * self.dim))
                    while not self.check_distances(heap_spawn, candidate):
                        if counter == self.max_tries:
                            over_tries = True
                            break
                        counter += 1
                        candidate = (int(random.random() * self.dim), 
                                     int(random.random() * self.dim), 
                                     int(random.random() * self.dim))

                    heap_spawn[i][0], heap_spawn[i][1], heap_spawn[i][2] = candidate[0], candidate[1], candidate[2]
                    if over_tries and heap_spawn[i][2] >= sum(self.map[candidate[0]][candidate[1]]):
                        heap_spawn[i][2] = (sum(self.map[candidate[0]][candidate[1]])) - 1

                for i in range(heap_number):
                    r1 = np.arange(start=math.ceil(heap_spawn[i][0] - self.heap_dimension_x / 2),
                                   stop=int(heap_spawn[i][0] + self.heap_dimension_x / 2) + 1, step=1)
                    r2 = np.arange(start=math.ceil(heap_spawn[i][1] - self.heap_dimension_y / 2),
                                   stop=int(heap_spawn[i][1] + self.heap_dimension_y / 2) + 1, step=1)
                    r3 = np.arange(start=math.ceil(heap_spawn[i][2] - self.heap_dimension_z / 2),
                                   stop=int(heap_spawn[i][2] + self.heap_dimension_z / 2) + 1, step=1)

                    r1 = [item for item in r1 if 0 <= item < self.dim]
                    r2 = [item for item in r2 if 0 <= item < self.dim]
                    r3 = [item for item in r3 if 0 <= item < self.dim]

                    indexes_to_remove = list()

                    for j in range(len(r1)):
                        for k in range(len(r2)):
                            for w in range(len(r3)):
                                if r3[w] > sum(self.map[r1[j]][r2[k]]):
                                    indexes_to_remove.append(w)

                    indexes_to_remove = list(set(indexes_to_remove))
                    indexes_to_remove.reverse()

                    for idx in indexes_to_remove:
                        r3.pop(idx)

                    for j in r1:
                        for k in r2:
                            for w in r3:
                                self.map[j][k][w] = 2

            self.map = np.rot90(self.map, axes=(1, 2))
