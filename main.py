import argparse
import time
import random

from src.config import config
from src.world_utils import World

from src.ant_utils import Ant, QueenAnt, WorkerAnt
from src.utils import all_workers_free


def parse_args():
    parser = argparse.ArgumentParser(description='Parser for model selection and pipeline selection')

    parser.add_argument('--2d', '--2D', help='Whether to use the 2D version of the script.'
                                             'Used for visualization', action='store_true', dest='two_dimensions',
                        default=False)

    parser.add_argument('--usage', help='Help on how to use the script', action='store_true')

    return parser.parse_args()


def make_assertions(cfg: dict):
    assert cfg['size'] >= 50, 'The minimum size of the map is 50, please set it accordingly'
    assert cfg['size'] <= 100000, 'This script uses some interpolation functions that will give a non-valid ' \
                                  'result for sizes higher than 100000'

    assert 1 < cfg['offset'] < cfg['size'] * 0.7, 'The offset should be lesser than the 70% of the size. ' \
                                                  'Please change the values in the config file'
    assert cfg['multiplier'] <= cfg['size'], 'The multiplier should not be greater than the size of the map. ' \
                                             'Please change the values in the config file'

    assert cfg['resource_placement'] in ['surface', 'normal'], 'The resource placement should be either ' \
                                                               '\'surface\' or \'normal\''

    assert cfg['worker_number'] > 0, 'The number of worker ants should be greater than 0'


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
    if config['use_seed']:
        random.seed(config['random_seed'])

    config['worker_number'] = int(config['worker_number'])

    print('Setting: ', end='')
    if config['2d']:
        print(f'2d world, mode:{config["resource_placement"]}, using {config["worker_number"]} workers ')
    else:
        print(f'3d world, mode:{config["resource_placement"]}, using {config["worker_number"]} workers ')

    start = time.time()  # to measure the time, performance study purposes
    world = World(config)
    world_generation_time = time.time() - start
    print(f'world_generation_time {world_generation_time}')

    world_type = '2d'
    if not config['2d']:
        world_type = '3d'

    Ant.set_hyperparameters(world_type=world_type, resource_spawn=config['resource_placement'])
    ants = list()

    # (0, 0) is the top left corner
    queen_spawn = world.generate_spawn_position()
    queen_ant = QueenAnt(size=config['size'], position=queen_spawn, max_iter=config['max_iter'])
    world.set_queen_ant_position(queen_ant.get_position())
    queen_ant.set_surroundings(world.get_surroundings('queen'))

    print('The worker ants spawned at positions: ', end='')

    for i in range(config['worker_number']):
        generated_position = world.generate_spawn_position(near_queen=True)
        print(generated_position, end=' ')
        ants.append(WorkerAnt(size=config['size'], position=generated_position, id=config['starting_id'] + i))
        world.register_ant_position(ants[i].get_id(), ants[i].get_position())

    print('')
    ants_spawn = [ant.get_spawn_position() for ant in ants]
    ants_spawn = list(set(ants_spawn))
    counter = 0
    val, reason = queen_ant.fully_explored(ants_spawn, counter)

    while not (val and all_workers_free(ants)):

        if counter % 1000 == 0 and counter > 0:
            print(f'{counter} iterations done')

        counter += 1
        world.update_pheromones()
        queen_ant.update_passed_time()
        for worker_ant in ants:

            world.update_ant_location(worker_ant.get_id(), worker_ant.get_position())
            worker_ant.set_surroundings(world.get_surroundings(worker_ant.get_id()))
            move, moved = worker_ant.move()
            worker_ant.reset_path_based_on_spawn()
            affected_ants = list()
            if moved:
                for second_worker_ant in ants:
                    if second_worker_ant != worker_ant:
                        if move == 'release_item':
                            if second_worker_ant.get_position() == (
                                    worker_ant.get_position()[0], worker_ant.get_position()[1] + 1,
                                    worker_ant.get_position()[2]):
                                affected_ants.append(second_worker_ant.get_id())
                                second_worker_ant.set_position(worker_ant.get_position(), side_effect=True)
                        else:
                            if second_worker_ant.get_position() == (
                                    worker_ant.get_position()[0], worker_ant.get_position()[1] - 1,
                                    worker_ant.get_position()[2]):
                                affected_ants.append(second_worker_ant.get_id())
                                second_worker_ant.set_position(worker_ant.get_position(), side_effect=True)

            pheromone = worker_ant.release_pheromones()
            world.update(worker_ant.get_id(), worker_ant.get_position(), move, pheromone, moved)
            if worker_ant.get_position() == worker_ant.get_spawn_position():
                worker_knowledge = worker_ant.pass_knowledge_to_queen()
                worker_map = worker_knowledge[0]
                worker_cycles = worker_knowledge[1]
                queen_ant.get_knowledge_from_worker(worker_map, worker_cycles)
                worker_ant.get_knowledge_from_queen(queen_ant.get_map())

        val, reason = queen_ant.fully_explored(ants_spawn, counter)

    # print(world)
    if reason == 'max_iter':
        print('Maximum iteration counter reached, unfortunately the stop criterion could not be reached '
              f'within {config["max_iter"]} iterations and it\'s assumed to be unreachable in this scenario')

    print(f'iteration counter {counter}')
    print('time', time.time() - start)


if __name__ == '__main__':
    main()
