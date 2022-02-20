import pickle
from drifter_ml import regression_tests
from drifter_ml.regression_tests import RegressionTests
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
import numpy as np
import pandas as pd




def test_mse():
    df = pd.read_csv("SeoulBikeData.csv")
    column_names = ['Date', 'Rented Bike Count', 'Hour', 'Temperature(°C)', 'Humidity(%)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons',
       'Holiday', 'Functioning Day', 'Year', 'Month', 'Day', 'DayName']
    target_name = "Rented Bike Count"
    reg = pickle.load(open("rf_model", "rb"))

    test_suite = RegressionTests(reg,
    df, target_name, column_names)
    mse_boundary = 15
    assert test_suite.mse_upper_boundary(mse_boundary)

def test_mae():
    df = pd.read_csv("SeoulBikeData.csv")
    column_names = ['Date', 'Rented Bike Count', 'Hour', 'Temperature(°C)', 'Humidity(%)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons',
       'Holiday', 'Functioning Day', 'Year', 'Month', 'Day', 'DayName']
    target_name = "Rented Bike Count"
    reg  = pickle.load(open("rf_model", "rb"))

    test_suite = RegressionTests(reg,
    df, target_name, column_names)
    mae_boundary = 10
    assert test_suite.mae_upper_boundary(mae_boundary)



def test_regression_basic():
    df, column_names, target_name, reg, _ = generate_regression_data_and_models()
    test_suite = regression_tests.RegressionTests(reg,
                                                  df,
                                                  target_name,
                                                  column_names)
    try:
        mse_upper_boundary = 10000
        mae_upper_boundary = 10000
        tse_upper_boundary = 10000
        tae_upper_boundary = 10000
        test_suite.upper_bound_regression_testing(
            mse_upper_boundary,
            mae_upper_boundary,
            tse_upper_boundary,
            tae_upper_boundary
        )
        assert True
    except:
        assert False