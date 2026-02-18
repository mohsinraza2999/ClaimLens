import pytest
from fastapi.testclient import TestClient
from src.inference.main import app

test_app=TestClient(app)
def health():

    response=test_app.get("/health")

    assert response.status_code == 200
    assert response.status=="ok"
    



