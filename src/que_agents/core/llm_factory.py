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
            config_path = "configs/llm_config.yaml"
            with open(config_path, "r") as f:
                cls._llm_config = yaml.safe_load(f)

    @classmethod
    def get_llm(cls, agent_type: str, **kwargs):
        cls._load_llm_config()

        provider_name = cls._llm_config["llm"]["default_provider"]
        provider_config = cls._llm_config["llm"]["providers"][provider_name]

        # Override temperature if specified for agent type
        temperature = cls._llm_config["llm"]["temperature"].get(
            agent_type, provider_config.get("temperature", 0.7)
        )

        model_name = kwargs.get("model_name", provider_config["models"]["default"])
        max_tokens = kwargs.get("max_tokens", provider_config.get("max_tokens"))
        timeout = kwargs.get("timeout", provider_config.get("timeout"))

        if provider_name == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY", provider_config["api_key"]),
                base_url=os.getenv("OPENAI_API_BASE", provider_config.get("api_base")),
                max_tokens=max_tokens,
                request_timeout=timeout,
            )
        elif provider_name == "groq":
            return ChatGroq(
                model_name=model_name,
                temperature=temperature,
                api_key=os.getenv("GROQ_API_KEY", provider_config["api_key"]),
                max_tokens=max_tokens,
                request_timeout=timeout,
            )
        elif provider_name == "anthropic":
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY", provider_config["api_key"]),
                base_url=os.getenv(
                    "ANTHROPIC_API_BASE", provider_config.get("api_base")
                ),
                max_tokens=max_tokens,
                request_timeout=timeout,
            )
        elif provider_name == "azure_openai":
            # Azure OpenAI requires specific setup
            return ChatOpenAI(
                azure_endpoint=os.getenv(
                    "AZURE_OPENAI_ENDPOINT", provider_config["api_base"]
                ),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", provider_config["api_key"]),
                azure_deployment=os.getenv(
                    "AZURE_OPENAI_DEPLOYMENT_NAME", provider_config["deployment_name"]
                ),
                api_version=provider_config["api_version"],
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=timeout,
            )
        elif provider_name == "local":
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=provider_config["api_base"],
                num_ctx=max_tokens,  # Ollama uses num_ctx for max_tokens
                request_timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    @classmethod
    def get_embedding_model(cls, model_name: str = "all-MiniLM-L6-v2"):
        # This is a placeholder. In a real scenario, you might have a separate factory
        # for embedding models or load them directly as needed.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
