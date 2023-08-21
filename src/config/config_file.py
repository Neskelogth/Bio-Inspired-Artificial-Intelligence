config = {

    'size': 100,
    'multiplier': 50,
    'offset': 50,

    'resource_placement': 'surface',
    # 'resource_placement': 'normal',
    'max_tries': 1000,

    'octaves': 3,
    'seed': 42,

    'heap_dimension_x': 5,
    'heap_dimension_y': 5,
    'heap_dimension_z': 5,
    'starting_id': 0,

    'cutoff_distance': 15,  # for 2d surface
    # 'cutoff_distance': 20,  # for 2d normal
    # 'cutoff_distance': 25,  # for 3d surface
    # 'cutoff_distance': 35,  # for 3d normal

    'worker_number': 5,
    'epsilon': 1e-6,

    'use_pheromones': True,
    'pheromone_duration': 50,
    'use_seed': True,
    'random_seed': 42,
    'max_iter': 1e6
}
