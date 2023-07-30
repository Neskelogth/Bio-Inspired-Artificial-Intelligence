from src.world_utils.world_generation import generate_surface, generate_surface_line, generate_2d_world, generate_world, \
    place_resources
import argparse
import time

from src.config_file import config


def parse_args():
    parser = argparse.ArgumentParser(description='Parser for model selection and pipeline selection')

    # parser.add_argument('pipeline', help='Type of pipeline to use, may be train, test or inference')
    parser.add_argument('--2d', '--2D', help='Whether to use the 2D version of the script. '
                                             'Used for visualization', action='store_true', dest='two_dimensions',
                        default=False)

    parser.add_argument('--usage', help='')

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

    if not cfg['2d']:
        assert not cfg['visualize'], 'The visualization is implemented only for 2d worlds, please either ' \
                                     'turn the visualization off or config the script to work with 2d worlds'


def main():
    args = parse_args()
    config['2d'] = args.two_dimensions

    make_assertions(config)

    start = time.time()
    if not config['2d']:
        surface = generate_surface(config['size'], multiplier=config['multiplier'], offset=config['offset'],
                                   octaves=config['octaves'], seed=config['seed'])
        world = generate_world(config['size'], surface)
    else:
        surface_line = generate_surface_line(config['size'], multiplier=config['multiplier'], offset=config['offset'],
                                             octaves=config['octaves'], seed=config['seed'])
        world = generate_2d_world(config['size'], surface_line)

    world = place_resources(world, config['resource_placement'], config['heap_dimension_x'],
                            config['heap_dimension_y'], config['heap_dimension_z'], config['cutoff_distance'])
    print('time', time.time() - start)


if __name__ == '__main__':
    main()
