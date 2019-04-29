import os


class Pooler:
    resolution = 14  # divide roi into 7*7
    scales = (0.0625,)
    sampling_scale = 0

class DataArgument:
    # Data Argumentation
    min_scale = 0.9
    max_scale = 1.1
    scale_num = 4
    pos_var = 3
    pos_num = 4






