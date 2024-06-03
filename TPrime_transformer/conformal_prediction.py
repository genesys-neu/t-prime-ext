import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import brentq

def false_negative_rate(prediction_set, gt_labels):
    return 1-((prediction_set * gt_labels).sum(axis=1)/gt_labels.sum(axis=1)).mean()



def conformal_prediction():
    pass


