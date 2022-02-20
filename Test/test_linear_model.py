from joblib import load, dump
import numpy as np
from Seoul_Bike_Prediction_Model import seoulbikedemand_prediction
from os import path
import sklearn
from sklearn.model_selection import train_test_split
import pytest
import math

def random_data_constructor(noise_mag=1.0):
    """
    Random data constructor utility for tests
    """
    num_points = 100
    X = 10*np.random.random(size=num_points)
    y = 2*X+3+2*noise_mag*np.random.normal(size=num_points)
    return X,y

#-------------------------------------------------------------------

def fixed_data_constructor():
    """
    Fixed data constructor utility for tests
    """
    num_points = 100
    X = np.linspace(1,10,num_points)
    y = 2*X+3
    return X,y

#-------------------------------------------------------------------

def test_model_return_object():
    """
    Tests the returned object of the modeling function
    """
    X,y = random_data_constructor()
    scores = seoulbikedemand_prediction(X,y)
    
    #=================================
    # TEST SUITE
    #=================================
    # Check the return object type
    assert isinstance(scores, dict)
    # Check the length of the returned object
    assert len(scores) == 2
    # Check the correctness of the names of the returned dict keys
    assert 'Train-score' in scores and 'Test-score' in scores

#-------------------------------------------------------------------