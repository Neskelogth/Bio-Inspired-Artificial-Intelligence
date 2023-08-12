import argparse
import time
# import random

# import matplotlib.pyplot as plt
# import numpy as np

from src.config.config_file import config
from src.world_utils.World import World

from src.ant_utils.Ant import Ant
# from src.ant_utils.QueenAnt import QueenAnt
from src.ant_utils.WorkerAnt import WorkerAnt


def parse_args():
    parser = argparse.ArgumentParser(description='Parser for model selection and pipeline selection')

    parser.add_argument('--2d', '--2D', help='Whether to use the 2D version of the script.'
                                             'Used for visualization', action='store_true', dest='two_dimensions',
                        default=False)

    parser.add_argument('--usage', help='Help on how to use the script', action='store_true')

    return parser.parse_args()


def make_assertions(cfg: dict):
    assert cfg['offset'] < cfg['size'] * 0.7, 'The offset should be lesser than the 70% of the size. ' \
                                              'Please change the values in the config file'
    assert cfg['multiplier'] <= cfg['size'], 'The multiplier should not be greater than the size of the map. ' \
                                             'Please change the values in the config file'
    assert cfg['resource_placement'] in ['surface', 'normal'], 'The resource placement should be either ' \
                                                               '\'surface\' or \'normal\''
    # assert cfg['size'] >= 50, 'The minimum size of the map is 50, please set it accordingly'  # Disabled for testing
    assert cfg['size'] <= 100000, 'This script uses some interpolation functions that will give a non-valid ' \
                                  'result for sizes higher than 100000'
    assert cfg['worker_number'] > 0, 'The number of worker ants should be greater than 0'

    # if not cfg['2d']:
    #     assert not cfg['visualize'], 'The visualization is implemented only for 2d worlds, please either ' \
    #                                  'turn the visualization off or config the script to work with 2d worlds'


def main():
    args = parse_args()

    if args.usage:
        print('This script let you simulate a world in which a queen ant and a variable number of worker ants are '
              'spawn in a randomly generated map, with the goal to explore it and gather resources. \n '
              'The script allows the following flags: \n '
              '\t --usage: will print this output and exit the program with a 0 exit status. \n '
              '\t --2d: will use the 2d version of the world. Using this world will make the script '
              'computationally lighter.')
        exit(0)

    config['2d'] = args.two_dimensions
    make_assertions(config)
    config['worker_number'] = int(config['worker_number'])

    print('Setting: ', end='')
    if config['2d']:
        print(f'2d world, mode:{config["resource_placement"]} ')
    else:
        print(f'3d world, mode:{config["resource_placement"]} ')

    start = time.time()  # to measure the time, performance study purposes
    world = World(config)
    world_generation_time = time.time() - start
    print(f'world_generation_time {world_generation_time}')

    world_type = '2d'
    if not config['2d']:
        world_type = '3d'

    Ant.set_hyperparameters(world_type=world_type, resource_spawn=config['resource_placement'])
    # ants = list()

    # # (0, 0) is the top left corner
    # queen_ant = QueenAnt(size=config['size'], position=(0, 0, 0))
    # world.set_queen_ant_position(queen_ant.get_position())
    worker_ant = WorkerAnt(size=config['size'], position=(5, 4, 0), id=config['starting_id'])

    world.register_ant_position(worker_ant.get_id(), worker_ant.get_position())
    worker_ant.set_surroundings(world.get_surroundings(worker_ant.get_id()))
    worker_ant.move()
    print(worker_ant.map)

    # print(worker_ant, '\n', worker_ant.map)
    print(world)
    # print(worker_ant.move())

    print('time', time.time() - start)


if __name__ == '__main__':
    main()
