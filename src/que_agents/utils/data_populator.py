# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module provides utilities for populating the database with sample data


import random
from datetime import datetime, timedelta

from src.que_agents.core.database import (
    AudienceSegment,
    CampaignMetrics,
    Customer,
    CustomerInteraction,
    KnowledgeBase,
    MarketingCampaign,
    MarketingPost,
    SmartDevice,
    SupportTicket,
    UserPreferences,
    get_session,
)


def populate_sample_data():
    session = get_session()

    try:
        # Sample Customers
        customers = [
            Customer(
                name="John Smith",
                email="john.smith@example.com",
                phone="+1-555-0101",
                company="TechCorp",
                tier="premium",
            ),
            Customer(
                name="Sarah Johnson",
                email="sarah.johnson@example.com",
                phone="+1-555-0102",
                company="StartupInc",
                tier="standard",
            ),
            Customer(
                name="Mike Chen",
                email="mike.chen@example.com",
                phone="+1-555-0103",
                company="Enterprise Ltd",
                tier="enterprise",
            ),
            Customer(
                name="Emily Davis",
                email="emily.davis@example.com",
                phone="+1-555-0104",
                company="SmallBiz",
                tier="standard",
            ),
            Customer(
                name="Robert Wilson",
                email="robert.wilson@example.com",
                phone="+1-555-0105",
                company="MegaCorp",
                tier="enterprise",
            ),
        ]

        for customer in customers:
            session.add(customer)
        session.commit()

        # Sample Knowledge Base Articles
        kb_articles = [
            KnowledgeBase(
                title="How to Reset Your Password",
                content="To reset your password: 1. Go to login page 2. Click 'Forgot Password' 3. Enter your email 4. Check your email for reset link 5. Follow the instructions in the email",
                category="account",
                tags=["password", "reset", "login", "account"],
            ),
            KnowledgeBase(
                title="Billing and Payment Issues",
                content="For billing questions: Check your account dashboard for current billing status. Contact support if you see unexpected charges. Payment methods can be updated in account settings.",
                category="billing",
                tags=["billing", "payment", "charges", "account"],
            ),
            KnowledgeBase(
                title="API Rate Limits",
                content="Our API has the following rate limits: Free tier: 100 requests/hour, Standard: 1000 requests/hour, Premium: 10000 requests/hour, Enterprise: Custom limits available.",
                category="technical",
                tags=["api", "rate", "limits", "technical"],
            ),
            KnowledgeBase(
                title="Data Export Procedures",
                content="To export your data: 1. Navigate to Settings > Data Export 2. Select data types to export 3. Choose format (CSV, JSON, XML) 4. Click 'Generate Export' 5. Download when ready",
                category="data",
                tags=["export", "data", "backup", "download"],
            ),
        ]

        for article in kb_articles:
            session.add(article)
        session.commit()

        # Sample Customer Interactions
        interactions = [
            CustomerInteraction(
                customer_id=1,
                interaction_type="chat",
                message="I can't log into my account",
                response="I can help you with that. Let me check your account status and guide you through the password reset process.",
                sentiment="neutral",
                satisfaction_score=4.5,
                agent_id="support_agent_1",
            ),
            CustomerInteraction(
                customer_id=2,
                interaction_type="email",
                message="When will the new features be available?",
                response="The new features are scheduled for release next month. I'll add you to our notification list for updates.",
                sentiment="positive",
                satisfaction_score=5.0,
                agent_id="support_agent_1",
            ),
            CustomerInteraction(
                customer_id=3,
                interaction_type="ticket",
                message="API is returning 500 errors",
                response="I've escalated this to our technical team. They're investigating the issue and will provide an update within 2 hours.",
                sentiment="negative",
                satisfaction_score=3.0,
                agent_id="support_agent_2",
            ),
        ]

        for interaction in interactions:
            session.add(interaction)
        session.commit()

        # Sample Support Tickets
        tickets = [
            SupportTicket(
                customer_id=1,
                title="Cannot access premium features",
                description="I upgraded to premium but still can't access the advanced analytics dashboard",
                category="billing",
                priority="high",
                status="in_progress",
                assigned_to="support_agent_1",
            ),
            SupportTicket(
                customer_id=3,
                title="API Integration Issues",
                description="Getting timeout errors when making API calls with large datasets",
                category="technical",
                priority="urgent",
                status="open",
                assigned_to="support_agent_2",
            ),
        ]

        for ticket in tickets:
            session.add(ticket)
        session.commit()

        # Sample Marketing Campaigns
        campaigns = [
            MarketingCampaign(
                name="Q1 Product Launch",
                description="Launch campaign for new AI features",
                campaign_type="social",
                status="active",
                target_audience={
                    "age_range": "25-45",
                    "interests": ["technology", "AI", "business"],
                },
                budget=10000.0,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now() + timedelta(days=30),
            ),
            MarketingCampaign(
                name="Customer Retention Email Series",
                description="Email series to improve customer retention",
                campaign_type="email",
                status="active",
                target_audience={
                    "tier": ["standard", "premium"],
                    "last_login": "30_days_ago",
                },
                budget=5000.0,
                start_date=datetime.now() - timedelta(days=15),
                end_date=datetime.now() + timedelta(days=45),
            ),
        ]

        for campaign in campaigns:
            session.add(campaign)
        session.commit()

        # Sample Audience Segments
        segments = [
            AudienceSegment(
                name="Tech Enthusiasts",
                description="Users interested in cutting-edge technology",
                criteria={
                    "interests": ["AI", "machine learning", "technology"],
                    "engagement": "high",
                },
                size=15000,
            ),
            AudienceSegment(
                name="Enterprise Customers",
                description="Large enterprise clients",
                criteria={"company_size": "1000+", "tier": "enterprise"},
                size=500,
            ),
            AudienceSegment(
                name="New Users",
                description="Users who signed up in the last 30 days",
                criteria={"signup_date": "last_30_days", "engagement": "low"},
                size=2500,
            ),
        ]

        for segment in segments:
            session.add(segment)
        session.commit()

        # Sample Marketing Posts
        posts = [
            MarketingPost(
                campaign_id=1,
                platform="twitter",
                content="🚀 Exciting news! Our new AI-powered analytics dashboard is now live. Get insights like never before! #AI #Analytics #Innovation",
                scheduled_time=datetime.now() + timedelta(hours=2),
                status="scheduled",
            ),
            MarketingPost(
                campaign_id=1,
                platform="linkedin",
                content="Transform your business with our latest AI features. Join thousands of companies already leveraging intelligent automation.",
                published_time=datetime.now() - timedelta(days=1),
                status="published",
            ),
        ]

        for post in posts:
            session.add(post)
        session.commit()

        # Sample Campaign Metrics
        for i in range(7):  # Last 7 days of metrics
            metric_date = datetime.now() - timedelta(days=i)
            metrics = [
                CampaignMetrics(
                    campaign_id=1,
                    metric_date=metric_date,
                    impressions=random.randint(1000, 5000),
                    clicks=random.randint(50, 200),
                    conversions=random.randint(5, 25),
                    cost=random.uniform(100, 500),
                    revenue=random.uniform(500, 2000),
                    engagement_rate=random.uniform(0.02, 0.08),
                    click_through_rate=random.uniform(0.01, 0.05),
                    conversion_rate=random.uniform(0.05, 0.15),
                ),
                CampaignMetrics(
                    campaign_id=2,
                    metric_date=metric_date,
                    impressions=random.randint(500, 2000),
                    clicks=random.randint(25, 100),
                    conversions=random.randint(2, 15),
                    cost=random.uniform(50, 250),
                    revenue=random.uniform(200, 1000),
                    engagement_rate=random.uniform(0.03, 0.10),
                    click_through_rate=random.uniform(0.02, 0.06),
                    conversion_rate=random.uniform(0.08, 0.20),
                ),
            ]

            for metric in metrics:
                session.add(metric)

        # Sample User Preferences
        user_preferences = [
            UserPreferences(
                user_id="1",
                preferences={
                    "music_genre": "classical",
                    "news_topics": ["tech", "AI"],
                    "preferred_language": "en",
                },
            ),
            UserPreferences(
                user_id="2",
                preferences={
                    "music_genre": "jazz",
                    "news_topics": ["finance", "politics"],
                    "preferred_language": "es",
                },
            ),
        ]
        for pref in user_preferences:
            session.add(pref)

        # Sample Smart Devices
        smart_devices = [
            SmartDevice(
                user_id="1",
                device_name="living room light",
                device_type="light",
                current_state={"power": "off", "brightness": 0},
                location="Living Room",
            ),
            SmartDevice(
                user_id="1",
                device_name="bedroom thermostat",
                device_type="thermostat",
                current_state={"power": "on", "temperature": 22.5, "unit": "celsius"},
                location="Bedroom",
            ),
            SmartDevice(
                user_id="2",
                device_name="kitchen speaker",
                device_type="speaker",
                current_state={"power": "on", "volume": 50, "playing": "none"},
                location="Kitchen",
            ),
        ]
        for device in smart_devices:
            session.add(device)

        session.commit()

        print("Sample data populated successfully!")
        print(f"Created {len(customers)} customers")
        print(f"Created {len(kb_articles)} knowledge base articles")
        print(f"Created {len(interactions)} customer interactions")
        print(f"Created {len(tickets)} support tickets")
        print(f"Created {len(campaigns)} marketing campaigns")
        print(f"Created {len(segments)} audience segments")
        print(f"Created {len(posts)} marketing posts")
        print(f"Created {14} campaign metrics entries")
        print(f"Created {len(user_preferences)} user preferences")
        print(f"Created {len(smart_devices)} smart devices")

    except Exception as e:
        session.rollback()
        print(f"Error populating data: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    populate_sample_data()
