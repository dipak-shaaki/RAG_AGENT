from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the RAG Agent API. Use /ingest to upload documents and /chat to interact."
    }

def test_read_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text

def test_read_redoc():
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "ReDoc" in response.text

def test_read_openapi():
    response = client.get("/openapi.json")
    assert response.status_code == 200

    payload = response.json()

    assert "openapi" in payload
    assert "info" in payload
    assert "paths" in payload
    assert "components" in payload
    assert "schemas" in payload["components"]
    assert "ChatRequest" in payload["components"]["schemas"]
    assert "ChatResponse" in payload["components"]["schemas"]
