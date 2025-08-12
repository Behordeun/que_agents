from unittest.mock import mock_open, patch

import pytest

from src.que_agents.utils.config_manager import ConfigManager


class TestConfigManager:
    """Test ConfigManager functionality"""

    def test_init_default(self):
        """Test ConfigManager initialization with defaults"""
        config_manager = ConfigManager()

        assert config_manager.config_dir == "configs"
        assert config_manager._cache == {}
        assert len(config_manager._config_paths) == 3

    def test_init_custom_dir(self):
        """Test ConfigManager initialization with custom directory"""
        config_manager = ConfigManager("custom_configs")

        assert config_manager.config_dir == "custom_configs"

    @patch(
        "builtins.open", new_callable=mock_open, read_data='api:\n  title: "Test API"'
    )
    @patch("os.path.exists")
    def test_load_config_success(self, mock_exists, mock_file):
        """Test successful configuration loading"""
        mock_exists.return_value = True

        config_manager = ConfigManager()
        config = config_manager.load_config("api_config.yaml")

        assert "api" in config
        assert config["api"]["title"] == "Test API"

    @patch(
        "builtins.open", new_callable=mock_open, read_data='api:\n  title: "Test API"'
    )
    @patch("os.path.exists")
    def test_load_config_with_cache(self, mock_exists, mock_file):
        """Test configuration loading with caching"""
        mock_exists.return_value = True

        config_manager = ConfigManager()

        # First load
        config1 = config_manager.load_config("api_config.yaml")
        # Second load (should use cache)
        config2 = config_manager.load_config("api_config.yaml")

        assert config1 == config2
        assert "api_config.yaml" in config_manager._cache

    @patch("os.path.exists")
    def test_load_config_file_not_found(self, mock_exists):
        """Test configuration loading when file doesn't exist"""
        mock_exists.return_value = False

        config_manager = ConfigManager()
        config = config_manager.load_config("nonexistent.yaml")

        # Should return fallback config
        assert isinstance(config, dict)

    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: content:")
    @patch("os.path.exists")
    def test_load_config_yaml_error(self, mock_exists, mock_file):
        """Test configuration loading with YAML parsing error"""
        mock_exists.return_value = True

        config_manager = ConfigManager()
        config = config_manager.load_config("invalid.yaml")

        # Should return fallback config
        assert isinstance(config, dict)

    @patch("builtins.open", side_effect=PermissionError("Access denied"))
    @patch("os.path.exists")
    def test_load_config_permission_error(self, mock_exists, mock_file):
        """Test configuration loading with permission error"""
        mock_exists.return_value = True

        config_manager = ConfigManager()
        config = config_manager.load_config("restricted.yaml")

        # Should return fallback config
        assert isinstance(config, dict)

    def test_validate_api_config(self):
        """Test API configuration validation"""
        config_manager = ConfigManager()

        config = {"api": {"title": "Custom Title"}}
        validated = config_manager._validate_api_config(config)

        assert validated["api"]["title"] == "Custom Title"
        assert "version" in validated["api"]
        assert "cors" in validated
        assert "authentication" in validated

    def test_validate_agent_config(self):
        """Test agent configuration validation"""
        config_manager = ConfigManager()

        config = {"customer_support": {"model_name": "custom-model"}}
        validated = config_manager._validate_agent_config(config)

        assert "customer_support" in validated
        assert "marketing" in validated
        assert "personal_virtual_assistant" in validated
        assert "financial_trading_bot" in validated

    def test_get_default_agent_config(self):
        """Test getting default agent configuration"""
        config_manager = ConfigManager()

        config = config_manager._get_default_agent_config("customer_support")

        assert "model_name" in config
        assert "temperature" in config
        assert "knowledge_base" in config
        assert "escalation_threshold" in config

    def test_get_fallback_config_api(self):
        """Test getting fallback API configuration"""
        config_manager = ConfigManager()

        config = config_manager._get_fallback_config("./configs/api_config.yaml")

        assert "api" in config
        assert "cors" in config
        assert "authentication" in config

    def test_get_fallback_config_agent(self):
        """Test getting fallback agent configuration"""
        config_manager = ConfigManager()

        config = config_manager._get_fallback_config("./configs/agent_config.yaml")

        assert "customer_support" in config
        assert "marketing" in config
        assert "personal_virtual_assistant" in config
        assert "financial_trading_bot" in config

    def test_get_fallback_config_unknown(self):
        """Test getting fallback configuration for unknown file"""
        config_manager = ConfigManager()

        config = config_manager._get_fallback_config("unknown.yaml")

        assert config == {}

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='customer_support:\n  model_name: "test-model"',
    )
    @patch("os.path.exists")
    def test_get_agent_config(self, mock_exists, mock_file):
        """Test getting specific agent configuration"""
        mock_exists.return_value = True

        config_manager = ConfigManager()
        config = config_manager.get_agent_config("customer_support")

        assert config is not None
        assert "model_name" in config

    @patch(
        "builtins.open", new_callable=mock_open, read_data='api:\n  title: "Test API"'
    )
    @patch("os.path.exists")
    def test_get_api_config(self, mock_exists, mock_file):
        """Test getting API configuration"""
        mock_exists.return_value = True

        config_manager = ConfigManager()
        config = config_manager.get_api_config()

        assert "api" in config

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_update_config_success(self, mock_makedirs, mock_exists, mock_file):
        """Test successful configuration update"""
        mock_exists.return_value = True

        config_manager = ConfigManager()
        config_manager._cache["test.yaml"] = {"existing": "config"}

        with patch.object(
            config_manager, "_find_writable_config_path", return_value="test.yaml"
        ):
            result = config_manager.update_config("test.yaml", {"new": "value"})

        assert result is True

    def test_update_config_no_writable_path(self):
        """Test configuration update when no writable path found"""
        config_manager = ConfigManager()

        with patch.object(
            config_manager, "_find_writable_config_path", return_value=None
        ):
            result = config_manager.update_config("test.yaml", {"new": "value"})

        assert result is False

    @patch("builtins.open", side_effect=Exception("Write error"))
    def test_update_config_write_error(self, mock_file):
        """Test configuration update with write error"""
        config_manager = ConfigManager()

        with patch.object(
            config_manager, "_find_writable_config_path", return_value="test.yaml"
        ):
            result = config_manager.update_config("test.yaml", {"new": "value"})

        assert result is False

    def test_deep_merge(self):
        """Test deep merge functionality"""
        config_manager = ConfigManager()

        base = {"a": {"b": 1, "c": 2}, "d": 3}
        update = {"a": {"b": 10, "e": 4}, "f": 5}

        config_manager._deep_merge(base, update)

        assert base["a"]["b"] == 10
        assert base["a"]["c"] == 2
        assert base["a"]["e"] == 4
        assert base["d"] == 3
        assert base["f"] == 5

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.remove")
    def test_find_writable_config_path_success(
        self, mock_remove, mock_file, mock_makedirs
    ):
        """Test finding writable configuration path"""
        config_manager = ConfigManager()

        path = config_manager._find_writable_config_path("test.yaml")

        assert path is not None
        assert "test.yaml" in path

    @patch("os.makedirs", side_effect=OSError("Permission denied"))
    def test_find_writable_config_path_failure(self, mock_makedirs):
        """Test finding writable configuration path with permission error"""
        config_manager = ConfigManager()

        path = config_manager._find_writable_config_path("test.yaml")

        assert path is None

    def test_clear_cache(self):
        """Test cache clearing"""
        config_manager = ConfigManager()
        config_manager._cache["test"] = {"data": "value"}

        config_manager.clear_cache()

        assert config_manager._cache == {}

    def test_get_cache_info(self):
        """Test getting cache information"""
        config_manager = ConfigManager()
        config_manager._cache["test1"] = {"data": "value1"}
        config_manager._cache["test2"] = {"data": "value2"}

        info = config_manager.get_cache_info()

        assert info["cache_size"] == 2
        assert "test1" in info["cached_files"]
        assert "test2" in info["cached_files"]
        assert "config_paths" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
