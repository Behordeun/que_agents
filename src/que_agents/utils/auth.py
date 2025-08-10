# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-10
# @Description: Authentication utilities for the API

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.que_agents.error_trace.errorlogger import system_logger
from src.que_agents.utils.config_manager import ConfigManager

# Security
security = HTTPBearer()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    config_manager: ConfigManager = Depends(),
) -> str:
    """Verify API token"""
    api_config = config_manager.get_api_config()
    expected_token = api_config["authentication"]["api_token"]

    INVALID_AUTH_TOKEN_MSG = "Invalid authentication token"

    if credentials.credentials != expected_token:
        system_logger.error(
            INVALID_AUTH_TOKEN_MSG,
            additional_info={"context": "Authentication"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INVALID_AUTH_TOKEN_MSG,
        )
    return credentials.credentials


def get_config_manager() -> ConfigManager:
    """Get config manager instance"""
    return ConfigManager()
