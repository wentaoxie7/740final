import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random

def reproduce():
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    """argparse handels type=bool in a weird way.
    See this stack overflow: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    we can use this function as type converter for boolean values
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')