import pytest
from fastapi.testclient import TestClient
from src.que_agents.api.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "Que Agents API" in response.text

def test_process_customer_message():
    response = client.post(
        "/customer-support/process",
        json={
            "customer_id": 1,
            "message": "I have a billing question about my last invoice."
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert "confidence" in response.json()
    assert "escalate" in response.json()

def test_create_campaign_plan():
    response = client.post(
        "/marketing/campaign-plan",
        json={
            "campaign_type": "lead_generation",
            "target_audience": "small business owners",
            "budget": 5000.0,
            "duration_days": 30,
            "goals": ["increase sign-ups", "expand market reach"],
            "channels": ["email", "social_media"],
            "content_requirements": ["email", "social_media"]
        }
    )
    assert response.status_code == 200
    assert "campaign_id" in response.json()
    assert "strategy" in response.json()
    assert "content_pieces" in response.json()

def test_search_knowledge_base():
    response = client.post(
        "/knowledge-base/search",
        json={
            "query": "password reset",
            "limit": 2
        }
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    # Add more assertions based on expected search results

def test_initialize_knowledge_base():
    response = client.post(
        "/knowledge-base/initialize"
    )
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Knowledge base initialized" in response.json()["message"]

# You might need to add more tests for edge cases, error handling, etc.


