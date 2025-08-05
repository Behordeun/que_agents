# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module contains the database models for the Que Agents application.

from datetime import datetime

import yaml
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


# Customer Support Agent Models
class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(50))
    company = Column(String(255))
    tier = Column(String(50), default="standard")  # standard, premium, enterprise
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    interactions = relationship("CustomerInteraction", back_populates="customer")
    tickets = relationship("SupportTicket", back_populates="customer")


class CustomerInteraction(Base):
    __tablename__ = "customer_interactions"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    interaction_type = Column(String(50), nullable=False)  # chat, email, phone, ticket
    message = Column(Text, nullable=False)
    response = Column(Text)
    sentiment = Column(String(20))  # positive, negative, neutral
    satisfaction_score = Column(Float)  # 1-5 rating
    agent_id = Column(String(100))  # AI agent identifier
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="interactions")


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100))  # technical, billing, general, etc.
    priority = Column(String(20), default="medium")  # low, medium, high, urgent
    status = Column(String(50), default="open")  # open, in_progress, resolved, closed
    assigned_to = Column(String(100))  # agent identifier
    resolution = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime)

    # Relationships
    customer = relationship("Customer", back_populates="tickets")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100))
    tags = Column(JSON)  # Array of tags for better searchability
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Marketing Agent Models
class MarketingCampaign(Base):
    __tablename__ = "marketing_campaigns"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    campaign_type = Column(String(50))  # email, social, ads, content
    status = Column(String(50), default="draft")  # draft, active, paused, completed
    target_audience = Column(JSON)  # Audience segmentation criteria
    budget = Column(Float)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    posts = relationship("MarketingPost", back_populates="campaign")
    metrics = relationship("CampaignMetrics", back_populates="campaign")


class MarketingPost(Base):
    __tablename__ = "marketing_posts"

    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("marketing_campaigns.id"), nullable=False)
    platform = Column(
        String(50), nullable=False
    )  # facebook, twitter, instagram, linkedin
    content = Column(Text, nullable=False)
    media_urls = Column(JSON)  # Array of media file URLs
    scheduled_time = Column(DateTime)
    published_time = Column(DateTime)
    status = Column(String(50), default="draft")  # draft, scheduled, published, failed
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    campaign = relationship("MarketingCampaign", back_populates="posts")


class AudienceSegment(Base):
    __tablename__ = "audience_segments"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    criteria = Column(JSON)  # Segmentation criteria (age, location, interests, etc.)
    size = Column(Integer)  # Estimated audience size
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CampaignMetrics(Base):
    __tablename__ = "campaign_metrics"

    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("marketing_campaigns.id"), nullable=False)
    metric_date = Column(DateTime, nullable=False)
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    revenue = Column(Float, default=0.0)
    engagement_rate = Column(Float, default=0.0)
    click_through_rate = Column(Float, default=0.0)
    conversion_rate = Column(Float, default=0.0)

    # Relationships
    campaign = relationship("MarketingCampaign", back_populates="metrics")


# Personal Assistant Agent Models
class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True, nullable=False)  # Assuming a generic user ID
    preferences = Column(
        JSON
    )  # Stores a dictionary of preferences (e.g., {"music_genre": "classical", "news_topics": ["tech", "science"]})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SmartDevice(Base):
    __tablename__ = "smart_devices"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)  # Owner of the device
    name = Column(
        String(255), nullable=False
    )  # e.g., "living room light", "bedroom thermostat"
    type = Column(String(50), nullable=False)  # e.g., "light", "thermostat", "speaker"
    status = Column(
        JSON
    )  # Current state of the device (e.g., {"power": "on", "brightness": 80})
    location = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Load database configuration
with open("configs/database_config.yaml", "r") as f:
    db_config = yaml.safe_load(f)

DATABASE_URL = db_config["database"]["url"]
ECHO_SQL = db_config["database"].get("echo", False)
POOL_SIZE = db_config["database"].get("pool_size", 10)
MAX_OVERFLOW = db_config["database"].get("max_overflow", 20)
POOL_TIMEOUT = db_config["database"].get("pool_timeout", 30)
POOL_RECYCLE = db_config["database"].get("pool_recycle", 3600)


def get_engine():
    return create_engine(
        DATABASE_URL,
        echo=ECHO_SQL,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        pool_recycle=POOL_RECYCLE,
    )


def get_session():
    engine = get_engine()
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_local()


def create_tables():
    """Create all tables in the database"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


if __name__ == "__main__":
    create_tables()
