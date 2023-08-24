config = {

    # should be between 50 and 100000,
    # but the suggestion is to keep it under 1000 due to the script's requirements in terms of memory
    'size': 50,
    'multiplier': 25,
    'offset': 25,

    'resource_placement': 'surface',
    # 'resource_placement': 'normal',
    'max_tries': 1000,

    'octaves': 3,
    'seed': 42,

    'heap_dimension_x': 5,
    'heap_dimension_y': 5,
    'heap_dimension_z': 5,

    'cutoff_distance': 15,

    'worker_number': 5,
    'starting_id': 0,  # id of the first worker ant

    'use_pheromones': True,
    'pheromone_duration': 200,

    'use_seed': False,
    'random_seed': 42,
    'max_iter': 1e9
}
