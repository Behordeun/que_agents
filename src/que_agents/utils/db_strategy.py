from sqlalchemy import inspect, text

from src.que_agents.core.database import get_engine, get_session
from src.que_agents.error_trace.errorlogger import system_logger

# Replace the connection string with your actual database URI
engine = get_engine()


def migrate_audience_segments_table():
    """Add missing columns to audience_segments table"""
    session = get_session()
    try:
        # Check current table structure
        inspector = inspect(engine)

        # Check if table exists
        if not inspector.has_table("audience_segments"):
            system_logger.info("Creating audience_segments table...")
            session.execute(
                text(
                    """
                CREATE TABLE audience_segments (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    criteria JSONB,
                    characteristics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
                )
            )
            session.commit()
            system_logger.info("✅ audience_segments table created successfully")
            return

        # Get existing columns
        columns = inspector.get_columns("audience_segments")
        column_names = [col["name"] for col in columns]
        system_logger.info(f"Existing columns: {column_names}")

        # Add missing columns
        if "characteristics" not in column_names:
            system_logger.info("Adding characteristics column...")
            session.execute(
                text(
                    """
                ALTER TABLE audience_segments 
                ADD COLUMN characteristics JSONB;
            """
                )
            )
            session.commit()
            system_logger.info("✅ characteristics column added")

        if "criteria" not in column_names:
            system_logger.info("Adding criteria column...")
            session.execute(
                text(
                    """
                ALTER TABLE audience_segments 
                ADD COLUMN criteria JSONB;
            """
                )
            )
            session.commit()
            system_logger.info("✅ criteria column added")

        # Update criteria column type if it exists but is wrong type
        if "criteria" in column_names:
            try:
                session.execute(
                    text(
                        """
                    ALTER TABLE audience_segments 
                    ALTER COLUMN criteria TYPE JSONB USING criteria::jsonb;
                """
                    )
                )
                session.commit()
                system_logger.info("✅ criteria column type updated to JSONB")
            except Exception as e:
                system_logger.info(
                    f"criteria column already correct type or update not needed: {e}"
                )
                session.rollback()

        # Add timestamps if missing
        if "created_at" not in column_names:
            session.execute(
                text(
                    """
                ALTER TABLE audience_segments 
                ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            """
                )
            )
            session.commit()
            system_logger.info("✅ created_at column added")

        if "updated_at" not in column_names:
            session.execute(
                text(
                    """
                ALTER TABLE audience_segments 
                ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            """
                )
            )
            session.commit()
            system_logger.info("✅ updated_at column added")

        system_logger.info("✅ All migrations completed successfully")

    except Exception as e:
        session.rollback()
        system_logger.error(
            f"❌ Error during migration: {e}",
            additional_info={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise e
    finally:
        session.close()


def migrate_marketing_campaigns_table():
    """Add missing columns to marketing_campaigns table"""
    session = get_session()
    try:
        inspector = inspect(engine)

        if not inspector.has_table("marketing_campaigns"):
            system_logger.info("Creating marketing_campaigns table...")
            session.execute(
                text(
                    """
                CREATE TABLE marketing_campaigns (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    campaign_type VARCHAR(100),
                    target_audience VARCHAR(255),
                    budget DECIMAL(15,2),
                    start_date DATE,
                    end_date DATE,
                    status VARCHAR(50) DEFAULT 'active',
                    strategy TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
                )
            )
            session.commit()
            system_logger.info("✅ marketing_campaigns table created successfully")
            return

        # Get existing columns
        columns = inspector.get_columns("marketing_campaigns")
        column_names = [col["name"] for col in columns]
        system_logger.info(f"Existing marketing_campaigns columns: {column_names}")

        # Add strategy column if missing
        if "strategy" not in column_names:
            system_logger.info("Adding strategy column...")
            session.execute(
                text(
                    """
                ALTER TABLE marketing_campaigns 
                ADD COLUMN strategy TEXT;
            """
                )
            )
            session.commit()
            system_logger.info("✅ strategy column added")

        system_logger.info("✅ Marketing campaigns table migration completed")

    except Exception as e:
        session.rollback()
        system_logger.error(
            f"❌ Error migrating marketing_campaigns: {e}",
            additional_info={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    system_logger.info("Starting database migrations...")

    try:
        migrate_audience_segments_table()
        migrate_marketing_campaigns_table()
        system_logger.info("🎉 All database migrations completed successfully!")
    except Exception as e:
        system_logger.error(
            f"❌ Migration failed: {e}",
            additional_info={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        exit(1)
