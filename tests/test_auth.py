from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from src.que_agents.utils.auth import (
    get_config_manager,
    get_token_from_state,
    verify_token,
)


class TestAuth:
    """Test authentication utilities"""

    @pytest.mark.asyncio
    @patch("src.que_agents.utils.auth.ConfigManager")
    async def test_verify_token_success(self, mock_config_manager_class):
        """Test successful token verification"""
        mock_config_manager = Mock()
        mock_config_manager.get_api_config.return_value = {
            "authentication": {"api_token": "valid_token"}
        }
        mock_config_manager_class.return_value = mock_config_manager

        request = Mock(spec=Request)
        request.state = Mock()
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="valid_token"
        )

        # Should not raise exception
        await verify_token(request, credentials, mock_config_manager)

        assert request.state.token == "valid_token"

    @pytest.mark.asyncio
    @patch("src.que_agents.utils.auth.ConfigManager")
    async def test_verify_token_invalid(self, mock_config_manager_class):
        """Test token verification with invalid token"""
        mock_config_manager = Mock()
        mock_config_manager.get_api_config.return_value = {
            "authentication": {"api_token": "valid_token"}
        }
        mock_config_manager_class.return_value = mock_config_manager

        request = Mock(spec=Request)
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_token"
        )

        with pytest.raises(HTTPException) as exc_info:
            await verify_token(request, credentials, mock_config_manager)

        assert exc_info.value.status_code == 401
        assert "Invalid authentication token" in exc_info.value.detail

    def test_get_token_from_state_success(self):
        """Test successful token extraction from header"""
        authorization = "Bearer valid_token_123"

        token = get_token_from_state(authorization)

        assert token == "valid_token_123"

    def test_get_token_from_state_missing_header(self):
        """Test token extraction with missing authorization header"""
        with pytest.raises(HTTPException) as exc_info:
            get_token_from_state(None)

        assert exc_info.value.status_code == 401
        assert "Authorization header missing" in exc_info.value.detail

    def test_get_token_from_state_invalid_scheme(self):
        """Test token extraction with invalid authorization scheme"""
        authorization = "Basic invalid_scheme"

        with pytest.raises(HTTPException) as exc_info:
            get_token_from_state(authorization)

        assert exc_info.value.status_code == 401
        assert "Invalid authorization scheme" in exc_info.value.detail

    def test_get_token_from_state_invalid_format(self):
        """Test token extraction with invalid header format"""
        authorization = "InvalidFormat"

        with pytest.raises(HTTPException) as exc_info:
            get_token_from_state(authorization)

        assert exc_info.value.status_code == 401
        assert "Invalid authorization header format" in exc_info.value.detail

    def test_get_config_manager(self):
        """Test config manager instance creation"""
        config_manager = get_config_manager()

        from src.que_agents.utils.config_manager import ConfigManager

        assert isinstance(config_manager, ConfigManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
