import math
import torch
import numpy as np
from scipy import spatial

def SetSeeds(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.set_printoptions(precision=8)
    np.random.seed(seed)
    
def compute_distance(predict, label):
    # 1. chebyshev
    chevb = spatial.distance.chebyshev(label, predict)

    # 2. clark
    clark = (predict - label) / (predict + label)
    clark = math.sqrt(np.sum(clark * clark))

    # 3. canberra
    canb = spatial.distance.canberra(label, predict)

    # 4. kldist
    kl = np.sum(label * np.log(label / predict)) 

    # 5.consine
    consine = 1- spatial.distance.cosine(label, predict)

    # 6.intersection
    inter = np.sum(np.minimum(label, predict))

    result = np.array([chevb, clark, canb, kl, consine, inter]).astype(float)
    
    return result