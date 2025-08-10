#!/usr/bin/env python3
# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Server startup script for the Agentic AI API

"""
Agentic AI API Server Startup Script

This script starts the FastAPI server with optimized settings for different environments.

Usage:
    python run_server.py --env development
    python run_server.py --env production
    python run_server.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys
from pathlib import Path

import uvicorn

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the Agentic AI API server")

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["development", "production", "testing"],
        default="development",
        help="Environment mode (default: development)",
    )

    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development only)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Logging level (default: info)",
    )

    return parser.parse_args()


def get_server_config(args):
    """Get server configuration based on environment"""
    base_config = {
        "app": "que_agents.api.main:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
    }

    if args.env == "development":
        return {
            **base_config,
            "reload": args.reload or True,
            "reload_dirs": ["src"],
            "access_log": True,
        }

    elif args.env == "production":
        return {
            **base_config,
            "reload": False,
            "workers": 4,
            "access_log": True,
            "use_colors": False,
        }

    elif args.env == "testing":
        return {
            **base_config,
            "reload": False,
            "access_log": False,
            "log_level": "warning",
        }

    return base_config


def main():
    """Main entry point"""
    args = parse_arguments()
    config = get_server_config(args)

    print("🚀 Starting Agentic AI API Server...")
    print(f"   Environment: {args.env}")
    print(f"   Host: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   Log Level: {config['log_level']}")

    if args.env == "development":
        print(f"   Auto-reload: {'enabled' if config.get('reload') else 'disabled'}")
    elif args.env == "production":
        print(f"   Workers: {config.get('workers', 1)}")

    print(f"   Docs available at: http://{config['host']}:{config['port']}/docs")
    print("=" * 60)

    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n🔄 Server shutting down...")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
