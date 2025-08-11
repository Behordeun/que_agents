# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module contains the database models for the Que Agents application.import os

from datetime import datetime

import yaml
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
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
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

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
    created_at = Column(DateTime, default=datetime.now)

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
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
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
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# Marketing Agent Models
class MarketingCampaign(Base):
    __tablename__ = "marketing_campaigns"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    campaign_type = Column(String, index=True)
    target_audience = Column(String)
    budget = Column(Float)
    start_date = Column(Date)
    end_date = Column(Date)
    status = Column(String, default="active")
    strategy = Column(Text)  # Add this field if it doesn't exist
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    posts = relationship("MarketingPost", back_populates="campaign")
    metrics = relationship("CampaignMetrics", back_populates="campaign")


class MarketingPost(Base):
    __tablename__ = "marketing_posts"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("marketing_campaigns.id"), nullable=False)
    platform = Column(String, nullable=False)
    content_type = Column(String)  # Add this field
    title = Column(String)
    content = Column(Text)
    hashtags = Column(JSON)  # Store as JSON array
    call_to_action = Column(String)
    estimated_reach = Column(Integer)
    scheduled_time = Column(DateTime)
    published_time = Column(DateTime)
    status = Column(String, default="draft")
    engagement_metrics = Column(JSON)  # Store engagement data
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    campaign = relationship("MarketingCampaign", back_populates="posts")


class AudienceSegment(Base):
    __tablename__ = "audience_segments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    criteria = Column(JSON)  # Store criteria as JSON
    characteristics = Column(
        JSON
    )  # Store characteristics including estimated_size as JSON
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


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


# Personal Virtual Assistant Agent Models
class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, unique=True)  # User identifier
    preferences = Column(JSON)  # User preferences (location, interests, etc.)
    learned_behaviors = Column(JSON)  # AI-learned user patterns
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    reminders = relationship("Reminder", back_populates="user")


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True)
    user_id = Column(
        String(100), ForeignKey("user_preferences.user_id"), nullable=False
    )
    title = Column(String(255), nullable=False)
    description = Column(Text)
    reminder_time = Column(DateTime, nullable=False)
    is_recurring = Column(Boolean, default=False)
    recurrence_pattern = Column(String(100))  # daily, weekly, monthly, etc.
    status = Column(String(50), default="active")  # active, completed, cancelled
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime)

    # Relationships
    user = relationship("UserPreferences", back_populates="reminders")


class SmartDevice(Base):
    __tablename__ = "smart_devices"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    device_name = Column(String(255), nullable=False)
    device_type = Column(
        String(100), nullable=False
    )  # light, thermostat, speaker, etc.
    location = Column(String(255))  # room or area
    current_state = Column(JSON)  # Current device state (on/off, temperature, etc.)
    capabilities = Column(JSON)  # What the device can do
    is_online = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PVAInteraction(Base):
    __tablename__ = "pva_interactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    intent = Column(String(100))  # weather, reminder, device_control, general_query
    user_message = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    entities_extracted = Column(JSON)  # Extracted entities from user message
    confidence_score = Column(Float)
    session_id = Column(String(100))  # For conversation tracking
    created_at = Column(DateTime, default=datetime.now)


# Financial Trading Bot Agent Models
class TradingStrategy(Base):
    __tablename__ = "trading_strategies"

    id = Column(Integer, primary_key=True)
    # Use the correct column name from the database
    strategy_type = Column(String(100), nullable=False)  # This exists
    description = Column(Text)
    parameters = Column(JSON)  # Strategy-specific parameters
    risk_parameters = Column(JSON)  # Risk management settings
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    trades = relationship("TradeLog", back_populates="strategy")


class TradeLog(Base):
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("trading_strategies.id"), nullable=False)
    symbol = Column(String(20), nullable=False)  # Stock/crypto symbol
    trade_type = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    market_conditions = Column(JSON)  # Market data at time of trade
    confidence_score = Column(Float)  # AI confidence in the trade
    executed_at = Column(DateTime, default=datetime.now)

    # Relationships
    strategy = relationship("TradingStrategy", back_populates="trades")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True)
    portfolio_name = Column(String(255), nullable=False, default="Default Portfolio")
    cash_balance = Column(Float, default=10000.0)  # Starting cash
    total_value = Column(Float, default=10000.0)  # Current total portfolio value
    holdings = Column(JSON)  # Current stock/crypto holdings
    performance_metrics = Column(JSON)  # ROI, Sharpe ratio, etc.
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float)
    market_cap = Column(Float)
    change_24h = Column(Float)  # 24-hour price change percentage
    timestamp = Column(DateTime, default=datetime.now)
    data_source = Column(String(100))  # API source
    additional_metrics = Column(JSON)  # RSI, MACD, etc.


class TradingSignal(Base):
    __tablename__ = "trading_signals"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(20), nullable=False)  # buy, sell, hold
    strength = Column(Float, nullable=False)  # Signal strength (0-1)
    strategy_source = Column(String(100))  # Which strategy generated the signal
    market_conditions = Column(JSON)  # Market data used for signal
    reasoning = Column(Text)  # AI reasoning for the signal
    created_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime)  # When signal becomes invalid


# Load database configuration
with open("./configs/database_config.yaml", "r") as f:
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
