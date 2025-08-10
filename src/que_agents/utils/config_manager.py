# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Configuration management module for the Agentic AI system

import os
from typing import Any, Dict, List, Optional

import yaml

from src.que_agents.error_trace.errorlogger import system_logger

AGENT_CONFIG_FILE = "agent_config.yaml"
API_CONFIG_FILE = "api_config.yaml"


class ConfigManager:
    """Configuration management with better error handling and validation"""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigManager with configurable directory

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self._cache = {}
        self._config_paths = self._get_config_paths()

    def _get_config_paths(self) -> List[str]:
        """Get potential configuration file paths in order of priority"""
        return [
            os.path.join(os.path.dirname(__file__), f"../../../{self.config_dir}"),
            self.config_dir,
            os.path.join(os.getcwd(), self.config_dir),
        ]

    def load_config(
        self, config_file: str = API_CONFIG_FILE, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration with fallback and caching

        Args:
            config_file: Name of the configuration file
            use_cache: Whether to use cached configuration

        Returns:
            Configuration dictionary
        """
        cache_key = config_file

        # Return cached config if available and requested
        if use_cache and cache_key in self._cache:
            system_logger.info(f"Using cached configuration for {config_file}")
            return self._cache[cache_key]

        config = self._load_config_from_file(config_file)

        # Cache the configuration
        if config:
            self._cache[cache_key] = config

        return config

    def _load_config_from_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file with multiple path attempts"""
        for config_dir in self._config_paths:
            config_path = os.path.join(config_dir, config_file)

            try:
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)

                    if config is None:
                        system_logger.warning(
                            f"Configuration file {config_path} is empty"
                        )
                        continue

                    if config_file == API_CONFIG_FILE:
                        return self._validate_api_config(config)
                    elif config_file == AGENT_CONFIG_FILE:
                        return self._validate_agent_config(config)
                    else:
                        return config
                        return config

            except yaml.YAMLError as e:
                system_logger.error(f"YAML parsing error in {config_path}: {str(e)}")
                continue
            except Exception as e:
                system_logger.warning(
                    f"Failed to load config from {config_path}: {str(e)}"
                )
                continue

        # Return fallback configuration based on file type
        system_logger.warning(f"Using fallback configuration for {config_file}")
        return self._get_fallback_config(config_file)

    def _validate_api_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance API configuration"""
        # Ensure required sections exist
        config.setdefault("api", {})
        config.setdefault("cors", {})
        config.setdefault("authentication", {})

        # Set API defaults
        api_defaults = {
            "title": "Agentic AI API",
            "description": "AI-powered customer support and marketing agents",
            "version": "1.0.0",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "host": "0.0.0.0",
            "port": 8000,
            "reload": True,
            "log_level": "info",
        }

        for key, value in api_defaults.items():
            config["api"].setdefault(key, value)

        # Set CORS defaults
        cors_defaults = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

        for key, value in cors_defaults.items():
            config["cors"].setdefault(key, value)

        # Set authentication defaults
        config["authentication"].setdefault("api_token", "demo-token-123")

        return config

    def _validate_agent_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance agent configuration"""
        required_agents = [
            "customer_support",
            "marketing",
            "personal_virtual_assistant",
            "financial_trading_bot",
        ]

        for agent in required_agents:
            if agent not in config:
                config[agent] = self._get_default_agent_config(agent)
                system_logger.warning(f"Added default configuration for {agent}")

        return config

    def _get_default_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get default configuration for specific agent"""
        defaults = {
            "model_name": "groq-llama3",
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30,
            "retry_attempts": 3,
        }

        agent_specific = {
            "customer_support": {
                "knowledge_base": "customer_support_kb",
                "escalation_threshold": 0.8,
            },
            "marketing": {
                "campaign_types": ["email", "social", "content"],
                "target_audience_analysis": True,
            },
            "personal_virtual_assistant": {
                "context_memory": True,
                "task_planning": True,
            },
            "financial_trading_bot": {
                "risk_tolerance": "moderate",
                "trading_strategy": "conservative",
                "market_analysis": True,
            },
        }

        config = defaults.copy()
        config.update(agent_specific.get(agent_name, {}))
        return config

    def _get_fallback_config(self, config_file: str) -> Dict[str, Any]:
        if config_file == API_CONFIG_FILE:
            return {
                "api": {
                    "title": "Agentic AI API",
                    "description": "AI-powered customer support and marketing agents",
                    "version": "1.0.0",
                    "docs_url": "/docs",
                    "redoc_url": "/redoc",
                    "host": "0.0.0.0",
                    "port": 8000,
                    "reload": True,
                    "log_level": "info",
                },
                "cors": {
                    "allow_origins": ["*"],
                    "allow_credentials": True,
                    "allow_methods": ["*"],
                    "allow_headers": ["*"],
                },
                "authentication": {"api_token": "demo-token-123"},
            }
        elif config_file == AGENT_CONFIG_FILE:
            return {
                "customer_support": self._get_default_agent_config("customer_support"),
                "marketing": self._get_default_agent_config("marketing"),
                "personal_virtual_assistant": self._get_default_agent_config(
                    "personal_virtual_assistant"
                ),
                "financial_trading_bot": self._get_default_agent_config(
                    "financial_trading_bot"
                ),
            }
        else:
            return {}

    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent"""
        agent_config = self.load_config(AGENT_CONFIG_FILE)
        return agent_config.get(agent_name)

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.load_config(API_CONFIG_FILE)

    def update_config(self, config_file: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration file with new values

        Args:
            config_file: Name of configuration file to update
            updates: Dictionary of updates to apply

        Returns:
            True if successful, False otherwise
        """
        try:
            current_config = self.load_config(config_file, use_cache=False)

            # Deep merge updates
            self._deep_merge(current_config, updates)

            # Find writable config path
            config_path = self._find_writable_config_path(config_file)
            if not config_path:
                system_logger.error(f"No writable path found for {config_file}")
                return False

            # Write updated configuration
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(current_config, f, default_flow_style=False, indent=2)

            # Update cache
            self._cache[config_file] = current_config

            system_logger.info(f"Configuration updated successfully: {config_path}")
            return True

        except Exception as e:
            system_logger.error(
                f"Failed to update configuration {config_file}: {str(e)}"
            )
            return False

    def _deep_merge(
        self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]
    ) -> None:
        """Deep merge two dictionaries"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    def _find_writable_config_path(self, config_file: str) -> Optional[str]:
        """Find a writable path for configuration file"""
        for config_dir in self._config_paths:
            try:
                # Create directory if it doesn't exist
                os.makedirs(config_dir, exist_ok=True)

                config_path = os.path.join(config_dir, config_file)

                # Test if we can write to this location
                test_path = os.path.join(config_dir, ".write_test")
                with open(test_path, "w") as f:
                    f.write("test")
                os.remove(test_path)

                return config_path

            except OSError:
                continue

        return None

    def clear_cache(self) -> None:
        """Clear configuration cache"""
        self._cache.clear()
        system_logger.info("Configuration cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached configurations"""
        return {
            "cached_files": list(self._cache.keys()),
            "cache_size": len(self._cache),
            "config_paths": self._config_paths,
        }
