import requests
import unittest
from fastapi.testclient import TestClient
from http import HTTPStatus

url = "https://share.streamlit.io/avangarde2225/seoul_bike_count_prediction/Seoul_Bike_Prediction_Model.py"
session = requests.Session()



class test_api_call(unittest.TestCase):
   
    def test_index():
        response = client.get(url)
        assert response.status_code == HTTPStatus.OK
        assert response.json()["message"] == HTTPStatus.OK.phrase
  
if __name__ == '__main__':
    unittest.main()
