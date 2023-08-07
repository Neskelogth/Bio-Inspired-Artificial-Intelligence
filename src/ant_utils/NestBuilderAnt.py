from .Ant import Ant


class NestBuilderAnt(Ant):
    def __init__(self, kwargs):
        super().__init__(type='nest_builder', **kwargs)


