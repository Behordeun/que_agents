#!/usr/bin/env python3
"""
Test script to verify marketing agent fixes
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.que_agents.agents.marketing_agent import MarketingAgent
    from src.que_agents.core.schemas import CampaignRequest, CampaignType, ContentType

    print("✅ Successfully imported required modules")

    # Test enum handling
    campaign_type = CampaignType.PRODUCT_LAUNCH
    print(f"✅ Campaign type enum: {campaign_type.value}")

    # Test creating a campaign request
    request = CampaignRequest(
        campaign_type=CampaignType.PRODUCT_LAUNCH,
        target_audience="tech professionals",
        budget=10000.0,
        duration_days=30,
        goals=["increase_awareness"],
        channels=["social_media"],
        content_requirements=[ContentType.SOCIAL_MEDIA],
    )
    print("✅ Successfully created CampaignRequest")

    # Test creating marketing agent
    try:
        agent = MarketingAgent()
        print("✅ Successfully created MarketingAgent")

        # Test campaign creation (this will test the database fixes)
        try:
            result = agent.create_marketing_campaign(
                {
                    "campaign_type": "product_launch",
                    "target_audience": "tech professionals",
                    "budget": 10000,
                    "duration_days": 30,
                    "goals": ["increase_awareness"],
                    "channels": ["social_media"],
                    "content_requirements": ["social_media"],
                }
            )

            if "error" not in result:
                print("✅ Successfully created marketing campaign")
                print(f"Campaign ID: {result.get('campaign_id', 'N/A')}")
            else:
                print(f"⚠️ Campaign creation returned error: {result['error']}")

        except Exception as campaign_error:
            print(f"❌ Campaign creation failed: {campaign_error}")

    except Exception as agent_error:
        print(f"❌ Agent creation failed: {agent_error}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n🔍 Test completed")
