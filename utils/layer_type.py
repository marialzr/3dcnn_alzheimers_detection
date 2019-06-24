from enum import Enum


class LayerType(Enum):
    CONVOLUTIONAL = 0
    POOLING = 1
    DENSE = 2
    DROPOUT = 3
