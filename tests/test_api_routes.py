import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.main import app

test_app=TestClient(app)
def health():

    response=test_app.get("/health")

    assert response.status_code == 200
    assert response.status=="ok"
    



