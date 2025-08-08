import logging
import sys
from pathlib import Path

from sqlalchemy import inspect, text

from src.que_agents.core.database import get_engine, get_session

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

engine = get_engine()

# Constant for timestamp default value
TIMESTAMP_DEFAULT_CURRENT = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"


def fix_database_schema():
    """Fix all database schema issues for marketing system"""
    session = get_session()

    try:
        logger.info("🔧 Starting comprehensive database schema fix...")

        # Fix marketing_campaigns table
        fix_marketing_campaigns_table(session)

        # Fix audience_segments table
        fix_audience_segments_table(session)

        logger.info("✅ All database schema fixes completed successfully!")

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Schema fix failed: {e}")
        raise e
    finally:
        session.close()


def _create_marketing_campaigns_table(session):
    logger.info("🆕 Creating marketing_campaigns table...")
    session.execute(
        text(
            f"""
        CREATE TABLE marketing_campaigns (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            target_audience TEXT,
            strategy TEXT,
            created_at {TIMESTAMP_DEFAULT_CURRENT},
            updated_at {TIMESTAMP_DEFAULT_CURRENT}
        );
        CREATE INDEX idx_marketing_campaigns_name ON marketing_campaigns(name);
    """
        )
    )
    session.commit()
    logger.info("✅ marketing_campaigns table created successfully")


def fix_marketing_campaigns_table(session):
    """Fix marketing_campaigns table schema"""
    logger.info("🔄 Fixing marketing_campaigns table...")
    inspector = inspect(engine)

    if not inspector.has_table("marketing_campaigns"):
        _create_marketing_campaigns_table(session)
        return

    columns = inspector.get_columns("marketing_campaigns")
    column_info = {col["name"]: col for col in columns}
    logger.info(f"📊 Current columns: {list(column_info.keys())}")

    _fix_target_audience_column(session, column_info)
    _add_missing_marketing_campaigns_columns(session, column_info)
    session.commit()
    logger.info("✅ marketing_campaigns table schema fixed successfully")


def _add_missing_marketing_campaigns_columns(session, column_info):
    required_columns = {
        "strategy": "TEXT",
        "created_at": TIMESTAMP_DEFAULT_CURRENT,
        "updated_at": TIMESTAMP_DEFAULT_CURRENT,
    }
    for col_name, col_type in required_columns.items():
        if col_name not in column_info:
            logger.info(f"➕ Adding missing column: {col_name}")
            session.execute(
                text(
                    f"""
                ALTER TABLE marketing_campaigns 
                ADD COLUMN {col_name} {col_type}
            """
                )
            )
            session.commit()
            logger.info(f"✅ {col_name} column added")


def _fix_target_audience_column(session, column_info):
    if "target_audience" not in column_info:
        return
    target_audience_type = str(column_info["target_audience"]["type"]).upper()
    logger.info(f"🔍 Current target_audience column type: {target_audience_type}")

    if "JSON" not in target_audience_type:
        return

    logger.info("🔄 Converting target_audience from JSON to TEXT...")
    backup_result = session.execute(
        text(
            """
        SELECT id, target_audience, name
        FROM marketing_campaigns 
        WHERE target_audience IS NOT NULL
    """
        )
    )
    existing_records = backup_result.fetchall()
    logger.info(f"📁 Found {len(existing_records)} existing records to preserve")

    session.execute(text("ALTER TABLE marketing_campaigns DROP COLUMN target_audience"))
    session.execute(
        text("ALTER TABLE marketing_campaigns ADD COLUMN target_audience TEXT")
    )

    for record in existing_records:
        campaign_id, target_audience_json, name = record
        text_value = _convert_target_audience_to_text(target_audience_json)
        session.execute(
            text(
                """
            UPDATE marketing_campaigns 
            SET target_audience = :text_value 
            WHERE id = :campaign_id
        """
            ),
            {"text_value": text_value, "campaign_id": campaign_id},
        )
        logger.info(f"✅ Restored data for campaign '{name}': {text_value}")

    session.commit()
    logger.info("✅ target_audience column converted to TEXT successfully")


def _convert_target_audience_to_text(target_audience_json):
    try:
        if isinstance(target_audience_json, dict):
            return str(target_audience_json.get("value", str(target_audience_json)))
        elif isinstance(target_audience_json, str):
            try:
                import json

                parsed = json.loads(target_audience_json)
                return (
                    str(parsed.get("value", parsed))
                    if isinstance(parsed, dict)
                    else str(parsed)
                )
            except Exception:
                return target_audience_json
        else:
            return str(target_audience_json)
    except Exception as e:
        logger.error(f"Error converting target_audience to text: {e}")
        return str(target_audience_json)


def fix_audience_segments_table(session):
    """Fix audience_segments table schema"""
    logger.info("🔄 Fixing audience_segments table...")

    inspector = inspect(engine)

    if not inspector.has_table("audience_segments"):
        session.execute(
            text(
                f"""
            CREATE TABLE audience_segments (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                criteria JSONB,
                characteristics JSONB,
                created_at {TIMESTAMP_DEFAULT_CURRENT},
                updated_at {TIMESTAMP_DEFAULT_CURRENT}
            );
            CREATE INDEX idx_audience_segments_name ON audience_segments(name);
        """
            )
        )
        session.commit()
        logger.info("✅ audience_segments table created successfully")
        return

    # Get current column info
    columns = inspector.get_columns("audience_segments")
    column_info = {col["name"]: col for col in columns}

    required_columns = {
        "characteristics": "JSONB",
        "criteria": "JSONB",
        "created_at": TIMESTAMP_DEFAULT_CURRENT,
        "updated_at": TIMESTAMP_DEFAULT_CURRENT,
    }

    for col_name, col_type in required_columns.items():
        if col_name not in column_info:
            logger.info(f"➕ Adding missing column: {col_name}")
            session.execute(
                text(
                    f"""
                ALTER TABLE audience_segments 
                ADD COLUMN {col_name} {col_type}
            """
                )
            )
            session.commit()
            logger.info(f"✅ {col_name} column added")


if __name__ == "__main__":
    fix_database_schema()
