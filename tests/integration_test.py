"""
Comprehensive integration test for the Agentic AI system
Tests end-to-end workflows and data integration
"""

import json
import time
from datetime import datetime

import requests
import yaml

# Load API configuration
with open("configs/api_config.yaml", "r") as f:
    api_config = yaml.safe_load(f)

# Configuration
API_BASE = f"http://{api_config['api']['host']}:{api_config['api']['port']}"
API_TOKEN = api_config["authentication"]["api_token"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}


def test_system_health():
    """Test system health and connectivity"""
    print("🔍 Testing System Health...")

    response = requests.get(f"{API_BASE}/health")
    assert response.status_code == 200

    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert "customer_support" in health_data["agents"]
    assert "marketing" in health_data["agents"]

    print("✅ System health check passed")
    return health_data


def test_knowledge_base_integration():
    """Test knowledge base search across different data types"""
    print("🔍 Testing Knowledge Base Integration...")

    # Test structured data search
    response = requests.get(
        f"{API_BASE}/api/v1/knowledge-base/search",
        headers=HEADERS,
        params={"query": "password reset", "limit": 5},
    )
    assert response.status_code == 200

    kb_data = response.json()
    assert len(kb_data["results"]) > 0

    # Verify different data types are represented
    source_types = {result["source_type"] for result in kb_data["results"]}
    print(f"   Found data types: {source_types}")

    print("✅ Knowledge base integration test passed")
    return kb_data


def test_customer_support_workflow():
    """Test complete customer support workflow"""
    print("🔍 Testing Customer Support Workflow...")

    # Test customer context retrieval
    response = requests.get(
        f"{API_BASE}/api/v1/customer-support/customer/1", headers=HEADERS
    )
    assert response.status_code == 200

    customer_data = response.json()
    assert "name" in customer_data
    assert "email" in customer_data

    # Test customer support chat
    chat_data = {
        "customer_id": 1,
        "message": "I need help with billing. My last invoice seems incorrect.",
    }

    response = requests.post(
        f"{API_BASE}/api/v1/customer-support/chat", headers=HEADERS, json=chat_data
    )
    assert response.status_code == 200

    support_response = response.json()
    assert "response" in support_response
    assert "confidence" in support_response
    assert "sentiment" in support_response
    assert isinstance(support_response["escalate"], bool)

    print(f"   Agent response confidence: {support_response['confidence']:.2f}")
    print(f"   Sentiment detected: {support_response['sentiment']}")
    print(f"   Escalation needed: {support_response['escalate']}")

    print("✅ Customer support workflow test passed")
    return support_response


def test_marketing_workflow():
    """Test complete marketing workflow"""
    print("🔍 Testing Marketing Workflow...")

    # Test campaign creation
    campaign_data = {
        "campaign_type": "lead_generation",
        "target_audience": "small business owners",
        "budget": 5000.0,
        "duration_days": 21,
        "goals": ["generate qualified leads", "increase brand awareness"],
        "channels": ["linkedin", "email"],
        "content_requirements": ["social_media", "email"],
    }

    response = requests.post(
        f"{API_BASE}/api/v1/marketing/campaign/create",
        headers=HEADERS,
        json=campaign_data,
    )
    assert response.status_code == 200

    campaign_response = response.json()
    assert "campaign_id" in campaign_response
    assert "strategy" in campaign_response
    assert "content_pieces" in campaign_response
    assert len(campaign_response["content_pieces"]) > 0

    print(f"   Campaign created: {campaign_response['campaign_id']}")
    print(f"   Content pieces generated: {len(campaign_response['content_pieces'])}")
    print(
        f"   Estimated reach: {campaign_response['estimated_performance']['total_reach']:,}"
    )

    # Test content generation
    content_data = {
        "platform": "linkedin",
        "content_type": "social_media",
        "campaign_theme": "Business Automation Solutions",
        "target_audience": "small business owners",
        "key_messages": ["save time", "increase efficiency", "reduce costs"],
    }

    response = requests.post(
        f"{API_BASE}/api/v1/marketing/content/generate",
        headers=HEADERS,
        json=content_data,
    )
    assert response.status_code == 200

    content_response = response.json()
    assert "content" in content_response
    assert "platform" in content_response
    assert len(content_response["hashtags"]) > 0

    print(f"   Content generated for: {content_response['platform']}")
    print(f"   Hashtags: {', '.join(content_response['hashtags'][:3])}...")

    print("✅ Marketing workflow test passed")
    return campaign_response, content_response


def test_database_integration():
    """Test database operations and data consistency"""
    print("🔍 Testing Database Integration...")

    # Test customer listing
    response = requests.get(
        f"{API_BASE}/api/v1/customers", headers=HEADERS, params={"limit": 10}
    )
    assert response.status_code == 200

    customers_data = response.json()
    assert "customers" in customers_data
    assert "total" in customers_data
    assert customers_data["total"] > 0

    # Test campaign listing
    response = requests.get(
        f"{API_BASE}/api/v1/campaigns", headers=HEADERS, params={"limit": 10}
    )
    assert response.status_code == 200

    campaigns_data = response.json()
    assert "campaigns" in campaigns_data
    assert "total" in campaigns_data

    print(f"   Total customers in database: {customers_data['total']}")
    print(f"   Total campaigns in database: {campaigns_data['total']}")

    print("✅ Database integration test passed")
    return customers_data, campaigns_data


def test_data_types_integration():
    """Test integration of structured, semi-structured, and unstructured data"""
    print("🔍 Testing Multi-Data Type Integration...")

    # Search for content that should span different data types
    test_queries = [
        "customer support",  # Should find unstructured docs
        "campaign performance",  # Should find semi-structured data
        "user account",  # Should find structured data
    ]

    data_types_found = set()

    for query in test_queries:
        response = requests.get(
            f"{API_BASE}/api/v1/knowledge-base/search",
            headers=HEADERS,
            params={"query": query, "limit": 10},
        )
        assert response.status_code == 200

        results = response.json()["results"]
        for result in results:
            data_types_found.add(result["source_type"])

    print(f"   Data types found across queries: {data_types_found}")

    # Verify we have multiple data types
    assert len(data_types_found) >= 2, "Should find multiple data types"

    print("✅ Multi-data type integration test passed")
    return data_types_found


def test_agent_collaboration():
    """Test scenarios where agents might work together"""
    print("🔍 Testing Agent Collaboration Scenarios...")

    # Scenario: Customer asks about marketing campaign they're part of
    support_request = {
        "customer_id": 1,
        "message": "I received an email about your new product launch campaign. Can you tell me more about it?",
    }

    response = requests.post(
        f"{API_BASE}/api/v1/customer-support/chat",
        headers=HEADERS,
        json=support_request,
    )
    assert response.status_code == 200

    support_response = response.json()

    # The support agent should be able to handle marketing-related queries
    assert len(support_response["response"]) > 50  # Should provide substantial response

    print(
        f"   Support agent handled marketing query with confidence: {support_response['confidence']:.2f}"
    )

    print("✅ Agent collaboration test passed")
    return support_response


def run_performance_test():
    """Basic performance test"""
    print("🔍 Running Performance Test...")

    start_time = time.time()

    # Make multiple concurrent-like requests
    for _ in range(5):
        response = requests.get(f"{API_BASE}/health")
        assert response.status_code == 200

    health_time = time.time() - start_time

    # Test API response time
    start_time = time.time()
    requests.get(
        f"{API_BASE}/api/v1/knowledge-base/search",
        headers=HEADERS,
        params={"query": "help", "limit": 5},
    )
    search_time = time.time() - start_time

    print(f"   Health check (5 requests): {health_time:.2f}s")
    print(f"   Knowledge base search: {search_time:.2f}s")

    # Basic performance assertions
    assert health_time < 5.0, "Health checks should be fast"
    assert search_time < 10.0, "Search should complete within reasonable time"

    print("✅ Performance test passed")


def generate_test_report(results):
    """Generate a comprehensive test report"""
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "system_status": "PASSED",
        "test_results": results,
        "summary": {
            "total_tests": len(results),
            "passed_tests": sum(
                1 for r in results.values() if r.get("status") == "PASSED"
            ),
            "failed_tests": sum(
                1 for r in results.values() if r.get("status") == "FAILED"
            ),
        },
    }

    return report


def main():
    """Run comprehensive integration tests"""
    print("=" * 60)
    print("🚀 AGENTIC AI SYSTEM - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)

    results = {}

    try:
        # Test 1: System Health
        health_data = test_system_health()
        results["system_health"] = {"status": "PASSED", "data": health_data}

        # Test 2: Knowledge Base Integration
        kb_data = test_knowledge_base_integration()
        results["knowledge_base"] = {"status": "PASSED", "data": kb_data}

        # Test 3: Customer Support Workflow
        support_data = test_customer_support_workflow()
        results["customer_support"] = {"status": "PASSED", "data": support_data}

        # Test 4: Marketing Workflow
        marketing_data = test_marketing_workflow()
        results["marketing"] = {"status": "PASSED", "data": marketing_data}

        # Test 5: Database Integration
        db_data = test_database_integration()
        results["database"] = {"status": "PASSED", "data": db_data}

        # Test 6: Multi-Data Type Integration
        data_types = test_data_types_integration()
        results["data_types"] = {"status": "PASSED", "data": list(data_types)}

        # Test 7: Agent Collaboration
        collab_data = test_agent_collaboration()
        results["collaboration"] = {"status": "PASSED", "data": collab_data}

        # Test 8: Performance
        run_performance_test()
        results["performance"] = {
            "status": "PASSED",
            "data": "Performance within acceptable limits",
        }

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)

        # Generate report
        report = generate_test_report(results)

        print("\n📊 TEST SUMMARY:")
        print(f"   Total Tests: {report['summary']['total_tests']}")
        print(f"   Passed: {report['summary']['passed_tests']}")
        print(f"   Failed: {report['summary']['failed_tests']}")
        print(
            f"   Success Rate: {(report['summary']['passed_tests']/report['summary']['total_tests']*100):.1f}%"
        )

        # Save report
        with open("integration_test_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print("\n📄 Detailed report saved to: integration_test_report.json")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
