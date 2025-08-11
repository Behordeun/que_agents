# @Author: Muhammad Abiodun SULAIMAN
# @Email: abiodun.msulaiman@gmail.com
# @Date: 2025-08-01 23:53:31
# @Last Modified by:   Muhammad Abiodun SULAIMAN
# @Last Modified time: 2025-08-01 23:53:31
# @Description: This module contains the LLMFactory class for creating language models based on configuration.

import os

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


class LLMFactory:
    _llm_config = None

    @classmethod
    def _load_llm_config(cls):
        if cls._llm_config is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs", "llm_config.yaml")
            with open(config_path, "r") as f:
                cls._llm_config = yaml.safe_load(f)

    @classmethod
    def get_llm(cls, agent_type: str, **kwargs):
        cls._load_llm_config()

        if cls._llm_config is None:
            raise RuntimeError("LLM configuration could not be loaded. Please check that 'llm_config.yaml' exists and is valid.")

        provider_name = cls._llm_config["llm"]["default_provider"]
        provider_config = cls._llm_config["llm"]["providers"][provider_name]

        temperature = cls._llm_config["llm"]["temperature"].get(
            agent_type, provider_config.get("temperature", 0.7)
        )

        model_name = kwargs.get("model_name", provider_config["models"]["default"])
        max_tokens = kwargs.get("max_tokens", provider_config.get("max_tokens"))
        timeout = kwargs.get("timeout", provider_config.get("timeout"))

        common_kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "request_timeout": timeout,
        }

        return cls._get_llm_by_provider(provider_name, provider_config, common_kwargs, max_tokens)

    @classmethod
    def _get_llm_by_provider(cls, provider_name, provider_config, common_kwargs, max_tokens):
        if provider_name == "openai":
            return cls._get_openai_llm(provider_config, common_kwargs)
        elif provider_name == "groq":
            return cls._get_groq_llm(provider_config, common_kwargs)
        elif provider_name == "anthropic":
            return cls._get_anthropic_llm(provider_config, common_kwargs)
        elif provider_name == "azure_openai":
            return cls._get_azure_openai_llm(provider_config, common_kwargs)
        elif provider_name == "local":
            return cls._get_local_llm(provider_config, common_kwargs, max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    @staticmethod
    def _get_openai_llm(provider_config, common_kwargs):
        api_key = provider_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        api_base = provider_config.get("api_base") or os.getenv("OPENAI_API_BASE")
        from pydantic import SecretStr
        return ChatOpenAI(
            **common_kwargs,
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=api_base,
        )

    @staticmethod
    def _get_groq_llm(provider_config, common_kwargs):
        api_key = provider_config.get("api_key") or os.getenv("GROQ_API_KEY")
        from pydantic import SecretStr
        return ChatGroq(
            **common_kwargs,
            api_key=SecretStr(api_key) if api_key is not None else None,
        )

    @staticmethod
    def _get_anthropic_llm(provider_config, common_kwargs):
        api_key = provider_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        api_base = provider_config.get("api_base") or os.getenv("ANTHROPIC_API_BASE")
        from pydantic import SecretStr
        if api_key is None:
            raise RuntimeError("Anthropic API key is required but not provided.")
        return ChatAnthropic(
            **common_kwargs,
            api_key=SecretStr(api_key),
            base_url=api_base,
        )

    @staticmethod
    def _get_azure_openai_llm(provider_config, common_kwargs):
        api_key = provider_config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        from pydantic import SecretStr
        _azure_deployment = provider_config.get("azure_deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        _api_version = provider_config.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION")
        return ChatOpenAI(
            **common_kwargs,
            api_key=SecretStr(api_key) if api_key is not None else None,
        )

    @staticmethod
    def _get_local_llm(provider_config, common_kwargs, max_tokens):
        return ChatOllama(
            **common_kwargs,
            base_url=provider_config["api_base"],
            num_ctx=max_tokens,  # Ollama uses num_ctx for max_tokens
        )

    @classmethod
    def get_embedding_model(cls, model_name: str = "all-MiniLM-L6-v2"):
        # This is a placeholder. In a real scenario, you might have a separate factory
        # for embedding models or load them directly as needed.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
