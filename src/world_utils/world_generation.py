from tqdm import tqdm
from perlin_noise import PerlinNoise
import numpy as np
import math
import random
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_surface(dim: int,
                     multiplier: int,
                     offset: int,
                     octaves: int = 3,
                     seed: int = 42):
    noise = PerlinNoise(octaves=octaves, seed=seed)
    x_pixel, y_pixel = dim, dim

    pic = list()

    for j in tqdm(range(x_pixel), desc='Generating surface level'):
        temp = list()
        for i in range(y_pixel):
            temp.append(math.ceil(noise([i / x_pixel, j / y_pixel]) * multiplier) + offset)
        pic.append(temp)

    return pic


def generate_surface_line(dim: int,
                          multiplier: int,
                          offset: int,
                          octaves: int = 3,
                          seed: int = 42):
    noise = PerlinNoise(octaves=octaves, seed=seed)
    x_pixel = dim

    pic = list()

    for i in tqdm(range(x_pixel)):
        pic.append(math.ceil(noise(i / x_pixel) * multiplier) + offset)

    return pic


def generate_2d_world(dim: int,
                      surface_line: list):
    world = np.zeros((dim, dim), dtype=np.int32)
    for i in range(len(surface_line)):
        world[i][:surface_line[i]] = np.array([1] * surface_line[i])

    return world


def generate_world(dim: int,
                   surface: list):
    world = np.zeros((dim, dim, dim), dtype=np.int32)
    print(world.shape)
    for i in range(dim):
        for j in range(dim):
            world[i][j][:surface[i][j]] = np.array([1] * surface[i][j])

    return world


def check_distances(world: np.ndarray | None,
                    positions: np.ndarray,
                    candidate: tuple | int,
                    mode: str,
                    cutoff_distance: int):
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

    # either surface mode for a 3d world or normal mode for 2d world
    elif positions.shape[1] == 2:

        for item in positions:
            if world is not None and candidate[1] >= sum(world[x]):
                return False

            if all(item == 0):
                continue

            if mode == 'surface':
                point = (item[0], item[1])
                candidate_point = (x, y)
            else:
                point = (item[0], item[1], 0)
                candidate_point = (x, y, 0)

    # normal mode for 3d world
    else:
        # positions.shape[1] == 3
        for item in positions:
            if world is not None and candidate[2] >= sum(world[x][y]):
                return False

            if all(item == 0):
                continue

            point = (item[0], item[1], item[2])
            candidate_point = (x, y, z)

    if (candidate_point is not None and point is not None and
            distance.euclidean(point, candidate_point) < cutoff_distance):
        return False

    return True


def place_resources(world: np.ndarray,
                    mode: str,
                    spawn_dim_x: int = 5,
                    spawn_dim_y: int = 5,
                    spawn_dim_z: int = 5,
                    max_tries: int = 100,
                    cutoff_distance: int = 15):

    size = world.shape[0]

    if len(world.shape) == 2:
        print('2d world, mode:', mode)

        # function to have 2 heaps at 50 size, 50 at 2500 size and 100 at 5000, polynomial interpolation
        heap_number = int((size ** 2 / 12127500) + 1567 * size / 80850 + 5000 / 4851)
        if mode == 'normal':
            heap_number *= 5

        if mode == 'surface':
            heap_spawn = np.zeros((heap_number,))
            for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                counter = 0
                candidate = int(random.random() * size)
                while not check_distances(None, heap_spawn, candidate, mode, cutoff_distance):
                    if counter == max_tries:
                        break
                    counter += 1
                    candidate = int(random.random() * size)
                heap_spawn[i] = candidate

            for i in range(heap_number):
                r = np.arange(start=math.ceil(heap_spawn[i] - spawn_dim_x / 2),
                              stop=int(heap_spawn[i] + spawn_dim_x / 2) + 1, step=1)

                r = [item for item in r if 0 <= item < size]

                for j in r:
                    surface_line_position = sum(world[j])
                    world[j][surface_line_position - 1] = 2

        else:
            heap_spawn = np.zeros((heap_number, 2))
            for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                counter = 0
                over_tries = False
                candidate = (int(random.random() * size), int(random.random() * size))
                while not check_distances(world, heap_spawn, candidate, mode, cutoff_distance):
                    if counter == max_tries:
                        over_tries = True
                        break
                    counter += 1
                    candidate = (int(random.random() * size), int(random.random() * size))

                heap_spawn[i][0], heap_spawn[i][1] = candidate[0], candidate[1]
                if over_tries and candidate[1] >= sum(world[candidate[0]]):
                    heap_spawn[i][1] = sum(world[candidate[0]]) - 1

            for i in tqdm(range(heap_number)):
                r1 = np.arange(start=math.ceil(heap_spawn[i][0] - spawn_dim_x / 2),
                              stop=int(heap_spawn[i][0] + spawn_dim_x / 2) + 1, step=1)
                r2 = np.arange(start=math.ceil(heap_spawn[i][1] - spawn_dim_y / 2),
                               stop=int(heap_spawn[i][1] + spawn_dim_y / 2) + 1, step=1)

                r1 = [item for item in r1 if 0 <= item < size]
                r2 = [item for item in r2 if 0 <= item < size]

                for j in r1:
                    for k in r2:
                        world[j][k] = 2

            plt.imshow(world)
            plt.show()

    else:
        print('3d world, mode:', mode)

        # function to have 100 heaps at 50 size, 5000 heaps at 1000 size,
        # 10000 at 2500 size and 100000 at 5000, polynomial interpolation
        heap_number = int(-500000000 / 549989 + (11099980 * size) / 549989 - size ** 2 / 13749725)
        if mode == 'normal':
            heap_number *= 20

        if mode == 'surface':
            heap_spawn = np.zeros((heap_number, 2))
            for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                counter = 0
                candidate = (int(random.random() * size), int(random.random() * size))
                while not check_distances(None, heap_spawn, candidate, mode, cutoff_distance):
                    if counter == max_tries:
                        break
                    counter += 1
                    candidate = (int(random.random() * size), int(random.random() * size))
                heap_spawn[i][0], heap_spawn[i][1] = candidate[0], candidate[1]

        else:
            heap_spawn = np.zeros((heap_number, 3))
            for i in tqdm(range(heap_number), desc='Computing resources spawn'):
                counter = 0
                over_tries = False
                candidate = (int(random.random() * size), int(random.random() * size), int(random.random() * size))
                while not check_distances(world, heap_spawn, candidate, mode, cutoff_distance):
                    if counter == max_tries:
                        over_tries = True
                        break
                    counter += 1
                    candidate = (int(random.random() * size), int(random.random() * size))

                heap_spawn[i][0], heap_spawn[i][1], heap_spawn[i][2] = candidate[0], candidate[1], candidate[2]
                if over_tries and heap_spawn[i][2] >= sum(sum(world[candidate[0]][candidate[1]])):
                    heap_spawn[i][2] = sum(sum(world[candidate[0]][candidate[1]])) - 1

    return world
