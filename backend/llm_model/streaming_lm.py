"""
Streaming Language Model module for DSPy integration.

This module provides a custom DSPy LM implementation that:
- Streams tokens from OpenAI in real-time
- Parses and formats thinking/reasoning content
- Queues content for async streaming
"""

import json
import logging
import re
import queue
from typing import Optional
import openai
import dspy

logger = logging.getLogger(__name__)


class StreamingLM(dspy.LM):
    """
    Custom DSPy Language Model with real-time streaming support.

    This LM streams tokens from OpenAI and formats them for display,
    distinguishing between thinking (reasoning) and response content.
    """

    def __init__(self, model: str, api_key: str, stream_queue: queue.Queue):
        """
        Initialize the streaming LM.

        Args:
            model: Model name (e.g., 'openai/gpt-4o-mini')
            api_key: OpenAI API key
            stream_queue: Queue for streaming content to async consumer
        """
        super().__init__(model=model, api_key=api_key)
        self.api_key = api_key
        self.stream_queue = stream_queue
        self.call_count = 0

    def parse_thinking_content(self, text: str) -> Optional[str]:
        """
        Parse thinking text and strip headers, return formatted content.

        Args:
            text: Raw thinking content with DSPy markers

        Returns:
            Formatted thinking content or None if empty
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
                sections.append(f"💭 Thought: {thought}")

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
                except:
                    sections.append(f"📋 Args: {args}")

        return '\n\n'.join(sections) if sections else None

    def stream_smoothly(self, text: str, msg_type: str, chunk_size: int = 3):
        """
        Stream text in small chunks for smooth output.

        Args:
            text: Text to stream
            msg_type: Type of message ('thinking', 'token', etc.)
            chunk_size: Number of characters per chunk
        """
        if not text:
            return

        for i in range(0, len(text), chunk_size):
            self.stream_queue.put((msg_type, text[i:i + chunk_size]))

    def is_thinking_call(self, prompt, messages) -> bool:
        """
        Detect if this is a thinking/reasoning call or final response.

        Args:
            prompt: The prompt text
            messages: The messages list

        Returns:
            True if this is a thinking call, False otherwise
        """
        prompt_text = str(prompt or messages)
        return (
            'next_thought' in prompt_text or
            'Thought' in prompt_text or
            self.call_count == 1
        )

    def extract_response_text(self, full_response: str) -> str:
        """
        Extract clean response text from various formats.

        Args:
            full_response: Raw LLM response

        Returns:
            Cleaned response text
        """
        extracted = None

        # Strategy 1: Extract from [[ ## response ## ]] markers
        if '[[ ## response ## ]]' in full_response:
            match = re.search(
                r'\[\[ ## response ## \]\](.*?)(?:\[\[ ## completed ## \]\]|$)',
                full_response,
                re.DOTALL
            )
            if match:
                extracted = match.group(1).strip()
                logger.info(f"Extracted via markers: {extracted[:100]}...")

        # Strategy 2: Extract from JSON format
        if not extracted:
            json_match = re.search(r'"response"\s*:\s*"([^"]+)"', full_response)
            if json_match:
                extracted = json_match.group(1)
                logger.info(f"Extracted via JSON: {extracted[:100]}...")

        # Strategy 3: Use full response as fallback
        if not extracted:
            extracted = full_response
            logger.info(f"Using full response: {extracted[:100]}...")

        return extracted

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Override DSPy's LM call to stream responses.

        Args:
            prompt: Prompt text (optional)
            messages: Message list (optional)
            **kwargs: Additional OpenAI parameters

        Returns:
            List containing the full response (for DSPy compatibility)
        """
        self.call_count += 1

        # Prepare the request
        if messages is None and prompt:
            messages = [{"role": "user", "content": prompt}]

        # Detect if this is a thinking call or final response
        is_thinking = self.is_thinking_call(prompt, messages)

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
                self.stream_smoothly(formatted, 'thinking')
        else:
            # Extract clean response text
            extracted = self.extract_response_text(full_response)
            # Stream the clean response
            self.stream_smoothly(extracted, 'token')

        # Return original response for DSPy
        return [full_response]
