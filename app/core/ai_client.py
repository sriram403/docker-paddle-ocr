"""
AI Client Abstraction Layer
Supports both OpenAI and Vertex AI Gemini Pro
"""
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import httpx
import openai
import google.auth
from google import genai

from app.core.config import (
    OPENAI_API_KEY,
    GCP_PROJECT_ID,
    GCP_LOCATION,
    VERTEX_MODEL
)
from app.core.logging import get_logger

logger = get_logger("ai_client")


class BaseAIClient(ABC):
    """Abstract base class for AI clients"""

    @abstractmethod
    def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text content from prompt"""
        pass

    @abstractmethod
    def generate_content_with_image(
        self,
        prompt: str,
        base64_image: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0
    ) -> str:
        """Generate content from prompt + image"""
        pass


class OpenAIClient(BaseAIClient):
    """OpenAI GPT-4 client implementation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
            self.client = None
            return

        custom_http_client = httpx.Client(
            http2=False,
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.client = openai.OpenAI(
            api_key=self.api_key,
            http_client=custom_http_client,
            max_retries=2
        )
        logger.info("OpenAI client initialized")

    def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        if not self.client:
            raise ValueError("OpenAI client not initialized (missing API key)")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model or "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def generate_content_with_image(
        self,
        prompt: str,
        base64_image: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0
    ) -> str:
        if not self.client:
            raise ValueError("OpenAI client not initialized (missing API key)")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        })

        kwargs = {
            "model": model or "gpt-4o",
            "messages": messages,
            "temperature": temperature
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class VertexAIClient(BaseAIClient):
    """Vertex AI Gemini Pro client implementation using ADC"""

    def __init__(self, project_id: str, location: str = "europe-west2"):
        self.project_id = project_id
        self.location = location

        if not self.project_id:
            logger.warning("GCP_PROJECT_ID not provided, Vertex AI client disabled")
            self.client = None
            return

        try:
            # Use Application Default Credentials (ADC)
            creds, adc_project = google.auth.default()
            logger.info(f"Using ADC. adc_project={adc_project}, target_project={project_id}")

            # Initialize Vertex AI client
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location
            )
            logger.info(f"Vertex AI client initialized (location={location})")

        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client: {e}")
            self.client = None

    def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        if not self.client:
            raise ValueError("Vertex AI client not initialized (check GCP_PROJECT_ID and credentials)")

        # Combine system prompt and user prompt for Gemini
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        model_name = model or VERTEX_MODEL

        try:
            # Gemini expects different config
            config = {"temperature": temperature}
            if max_tokens:
                config["max_output_tokens"] = max_tokens

            # For JSON mode, add instruction to the prompt
            if response_format == "json":
                full_prompt = f"{full_prompt}\n\nRespond only with valid JSON."

            response = self.client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=config
            )

            return response.text

        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
            raise

    def generate_content_with_image(
        self,
        prompt: str,
        base64_image: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0
    ) -> str:
        if not self.client:
            raise ValueError("Vertex AI client not initialized (check GCP_PROJECT_ID and credentials)")

        import base64

        # Combine prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        if response_format == "json":
            full_prompt = f"{full_prompt}\n\nRespond only with valid JSON."

        model_name = model or VERTEX_MODEL

        try:
            # Decode base64 image for Gemini
            image_bytes = base64.b64decode(base64_image)

            # Gemini multimodal format
            contents = [
                full_prompt,
                {"mime_type": "image/jpeg", "data": base64_image}
            ]

            config = {"temperature": temperature}

            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )

            return response.text

        except Exception as e:
            logger.error(f"Vertex AI image generation failed: {e}")
            raise


class AIClientFactory:
    """Factory to create appropriate AI client based on provider"""

    @staticmethod
    def create(provider: str = "openai") -> BaseAIClient:
        """
        Create AI client based on provider choice

        Args:
            provider: "openai" or "vertex"

        Returns:
            Configured AI client instance
        """
        provider = provider.lower().strip()

        if provider == "openai":
            logger.info("Creating OpenAI client")
            return OpenAIClient(api_key=OPENAI_API_KEY)

        elif provider == "vertex":
            logger.info("Creating Vertex AI client")
            return VertexAIClient(
                project_id=GCP_PROJECT_ID,
                location=GCP_LOCATION
            )

        else:
            raise ValueError(f"Unknown AI provider: {provider}. Use 'openai' or 'vertex'")
