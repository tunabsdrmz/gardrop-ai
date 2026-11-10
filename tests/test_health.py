from app.config import Settings, get_settings
from app.main import app


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stylist_enabled_when_key_is_set(client):
    app.dependency_overrides[get_settings] = lambda: Settings(anthropic_api_key="test-key")

    assert client.get("/health").json()["stylist_enabled"] is True


def test_unknown_route_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_api_key_is_required_when_configured(client):
    app.dependency_overrides[get_settings] = lambda: Settings(service_api_key="secret")
    body = {"item": {"id": "1", "name": "Tee", "category": "top"}, "closet": []}

    assert client.post("/match", json=body).status_code == 401
    assert client.post("/match", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200
