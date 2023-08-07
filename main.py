import argparse
import time

# import matplotlib.pyplot as plt

from src.config_file import config
from src.world_utils.World import World

from src.ant_utils.Ant import Ant
from src.ant_utils.QueenAnt import QueenAnt
from src.ant_utils.WorkerAnt import WorkerAnt


def parse_args():
    parser = argparse.ArgumentParser(description='Parser for model selection and pipeline selection')

    # parser.add_argument('pipeline', help='Type of pipeline to use, may be train, test or inference')
    parser.add_argument('--2d', '--2D', help='Whether to use the 2D version of the script. '
                                             'Used for visualization', action='store_true', dest='two_dimensions',
                        default=False)

    parser.add_argument('--usage', help='Help on how to use the script')

    return parser.parse_args()


def make_assertions(cfg: dict):
    assert cfg['offset'] < cfg['size'] * 0.7, 'The offset should be lesser than the 70% of the size. ' \
                                              'Please change the values in the config file'
    assert cfg['multiplier'] <= cfg['size'], 'The multiplier should not be greater than the size of the map. ' \
                                             'Please change the values in the config file'
    assert cfg['resource_placement'] in ['surface', 'normal'], 'The resource placement should be either ' \
                                                               '\'surface\' or \'normal\''
    assert cfg['size'] >= 50, 'The minimum size of the map is 50, please set it accordingly'
    assert cfg['size'] <= 100000, 'This script uses some interpolation functions that will give a non-valid ' \
                                  'result for sizes higher than 100000'
    assert cfg['worker_number'] > 0, 'The number of worker ants should be greater than 0'

    # if not cfg['2d']:
    #     assert not cfg['visualize'], 'The visualization is implemented only for 2d worlds, please either ' \
    #                                  'turn the visualization off or config the script to work with 2d worlds'


def main():
    args = parse_args()

    # TODO usage param
    if args.usage:
        # print usage
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
    # world = World(config)
    world_generation_time = time.time() - start
    print(f'world_generation_time {world_generation_time}')

    world_type = '2d'
    if not config['2d']:
        world_type = '3d'

    Ant.set_hyperparameters(world_type=world_type, resource_spawn=config['resource_placement'])
    ants = list()

    ants.append(QueenAnt(size=config['size'], position=(0, 0, 0)))
    print(ants[0])
    ants.append(WorkerAnt(size=config['size'], position=(1, 0, 0)))
    print(ants[1])

    print('time', time.time() - start)


if __name__ == '__main__':
    main()
