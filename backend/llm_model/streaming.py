"""Streaming functionality for real-time LLM responses."""
import json
import logging
import re
import dspy
import openai
from openai import AsyncOpenAI
from typing import Any
from .config import Config
from .utils import format_json_safely

logger = logging.getLogger(__name__)


class StreamingLM(dspy.LM):
    """Custom DSPy Language Model with streaming support."""

    def __init__(self, model: str, api_key: str, stream_queue: Any):
        """
        Initialize the streaming language model.

        Args:
            model: Model identifier (e.g., 'openai/gpt-4o-mini')
            api_key: OpenAI API key
            stream_queue: Queue for streaming tokens to async generator
        """
        super().__init__(model=model, api_key=api_key)
        self.api_key = api_key
        self.stream_queue = stream_queue
        self.base_client = AsyncOpenAI(api_key=api_key)
        self.call_count = 0  # Track which LM call we're on

    def parse_thinking_content(self, text: str) -> str:
        """
        Parse thinking text and strip headers, return formatted content.

        Extracts and formats:
        - next_thought: The agent's reasoning
        - next_tool_name: Tool being invoked
        - next_tool_args: Arguments for the tool

        Args:
            text: Raw thinking text with DSPy markers

        Returns:
            Formatted thinking content or None if no content found
        """
        sections = []

        # Extract next_thought
        thought_match = re.search(
            r'\[\[ ## next_thought ## \]\](.*?)(?=\[\[|$)',
            text,
            re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()
            if thought:
                sections.append(f"💭 {thought}")

        # Extract next_tool_name
        tool_match = re.search(
            r'\[\[ ## next_tool_name ## \]\](.*?)(?=\[\[|$)',
            text,
            re.DOTALL
        )
        if tool_match:
            tool = tool_match.group(1).strip().strip("'\"")
            if tool and tool != 'finish':
                sections.append(f"🔧 Tool: {tool}")

        # Extract next_tool_args
        args_match = re.search(
            r'\[\[ ## next_tool_args ## \]\](.*?)(?=\[\[|$)',
            text,
            re.DOTALL
        )
        if args_match:
            args = args_match.group(1).strip()
            if args and args != '{}':
                try:
                    args_obj = json.loads(args)
                    sections.append(f"📋 Args: {json.dumps(args_obj, indent=2)}")
                except (json.JSONDecodeError, ValueError, TypeError):
                    sections.append(f"📋 Args: {args}")

        return '\n\n'.join(sections) if sections else None

    def extract_response_content(self, text: str) -> str:
        """
        Extract clean response text from LLM output.

        Tries multiple extraction strategies:
        1. Extract from [[ ## response ## ]] markers
        2. Extract from JSON format
        3. Use full response as fallback

        Args:
            text: Raw LLM response

        Returns:
            Extracted response text
        """
        # Strategy 1: Extract from [[ ## response ## ]] markers
        if '[[ ## response ## ]]' in text:
            match = re.search(
                r'\[\[ ## response ## \]\](.*?)(?:\[\[ ## completed ## \]\]|$)',
                text,
                re.DOTALL
            )
            if match:
                extracted = match.group(1).strip()
                logger.info(f"Extracted via markers: {extracted[:100]}...")
                return extracted

        # Strategy 2: Extract from JSON format
        json_match = re.search(r'"response"\s*:\s*"([^"]+)"', text)
        if json_match:
            extracted = json_match.group(1)
            logger.info(f"Extracted via JSON: {extracted[:100]}...")
            return extracted

        # Strategy 3: Use full response as fallback
        logger.info(f"Using full response: {text[:100]}...")
        return text

    def stream_text_smoothly(self, text: str, msg_type: str) -> None:
        """
        Stream text in small chunks for smooth output.

        Args:
            text: Text to stream
            msg_type: Message type ('token', 'thinking', etc.)
        """
        if not text:
            return

        chunk_size = Config.STREAM_CHUNK_SIZE
        for i in range(0, len(text), chunk_size):
            self.stream_queue.put((msg_type, text[i:i + chunk_size]))

    def __call__(self, prompt: str = None, messages: list = None, **kwargs) -> list:
        """
        Override DSPy LM call to capture and stream responses.

        Args:
            prompt: Optional prompt string
            messages: Optional list of message dicts
            **kwargs: Additional arguments for OpenAI API

        Returns:
            List containing the full response (for DSPy compatibility)
        """
        self.call_count += 1

        # Prepare the request
        if messages is None and prompt:
            messages = [{"role": "user", "content": prompt}]

        # Detect if this is a thinking call or final response
        prompt_text = str(prompt or messages)
        is_thinking = (
            'next_thought' in prompt_text or
            'Thought' in prompt_text or
            self.call_count == 1
        )

        # Make streaming request to OpenAI
        client = openai.OpenAI(api_key=self.api_key)
        stream = client.chat.completions.create(
            model=self.model.replace('openai/', ''),
            messages=messages,
            stream=True,
            **kwargs
        )

        # Collect full response while streaming from OpenAI
        full_response = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content

        logger.info(f"=== LM Call {self.call_count} ===")
        logger.info(f"Is Thinking: {is_thinking}")
        logger.info(f"Response Length: {len(full_response)} chars")
        logger.info(f"Response: {full_response[:300]}...")
        logger.info("=" * 50)

        # Process and stream based on type
        if is_thinking:
            # Parse thinking content
            formatted = self.parse_thinking_content(full_response)
            if formatted:
                self.stream_text_smoothly(formatted, 'thinking')
        else:
            # Extract clean response text
            extracted = self.extract_response_content(full_response)
            # Stream the clean response
            self.stream_text_smoothly(extracted, 'token')

        # Return original response for DSPy
        return [full_response]
