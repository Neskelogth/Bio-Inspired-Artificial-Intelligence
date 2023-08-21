def all_workers_free(ants):
    for worker in ants:
        if worker.get_picked_item() is not None:
            return False

    return True
