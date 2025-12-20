"""
Streaming Language Model module for DSPy integration.

This module provides a custom DSPy LM implementation that:
- Streams tokens from OpenAI in real-time
- Parses and formats thinking/reasoning content
- Queues content for async streaming
- Captures complete raw outputs for training data generation
"""

import json
import logging
import os
import re
import queue
from datetime import datetime
from typing import Optional, List, Dict, Any
import openai
import dspy

logger = logging.getLogger(__name__)

# Enable verbose logging via environment variable
VERBOSE_RAW_OUTPUT = os.getenv('VERBOSE_RAW_OUTPUT', 'false').lower() == 'true'
LOG_RAW_TO_FILE = os.getenv('LOG_RAW_TO_FILE', 'false').lower() == 'true'
RAW_OUTPUT_LOG_DIR = os.getenv('RAW_OUTPUT_LOG_DIR', 'logs/raw_outputs')


class StreamingLM(dspy.LM):
    """
    Custom DSPy Language Model with real-time streaming support.

    This LM streams tokens from OpenAI and formats them for display,
    distinguishing between thinking (reasoning) and response content.
    """

    def __init__(self, model: str, api_key: str, stream_queue: queue.Queue,
                 capture_for_training: bool = False):
        """
        Initialize the streaming LM.

        Args:
            model: Model name (e.g., 'openai/gpt-4o-mini')
            api_key: OpenAI API key
            stream_queue: Queue for streaming content to async consumer
            capture_for_training: If True, capture all raw outputs for training data
        """
        super().__init__(model=model, api_key=api_key)
        self.api_key = api_key
        self.stream_queue = stream_queue
        self.call_count = 0
        self.capture_for_training = capture_for_training
        self.captured_outputs: List[Dict[str, Any]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory if file logging is enabled
        if LOG_RAW_TO_FILE:
            os.makedirs(RAW_OUTPUT_LOG_DIR, exist_ok=True)

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

    def _log_raw_output(self, call_num: int, is_thinking: bool, prompt: Any,
                        messages: Any, full_response: str, parsed: Dict[str, Any]):
        """
        Log the complete raw output for debugging and training data capture.

        Args:
            call_num: The LM call number in this session
            is_thinking: Whether this was a thinking/reasoning call
            prompt: Original prompt
            messages: Original messages
            full_response: Complete raw LLM output
            parsed: Parsed components (thought, tool_name, tool_args)
        """
        separator = "=" * 80
        header = f"{'🧠 THINKING CALL' if is_thinking else '💬 RESPONSE CALL'} #{call_num}"

        # Build detailed log output
        log_lines = [
            "",
            separator,
            header,
            separator,
            "",
            "📥 INPUT:",
            "-" * 40,
        ]

        # Log the input (truncate if very long)
        input_str = json.dumps(messages if messages else prompt, indent=2, default=str)
        if len(input_str) > 2000 and not VERBOSE_RAW_OUTPUT:
            log_lines.append(input_str[:2000] + "\n... [truncated, set VERBOSE_RAW_OUTPUT=true for full output]")
        else:
            log_lines.append(input_str)

        log_lines.extend([
            "",
            "📤 RAW OUTPUT:",
            "-" * 40,
            full_response,
            "",
            "🔍 PARSED COMPONENTS:",
            "-" * 40,
            f"  next_thought: {parsed.get('thought', 'N/A')[:200]}..." if parsed.get('thought') else "  next_thought: N/A",
            f"  next_tool_name: {parsed.get('tool_name', 'N/A')}",
            f"  next_tool_args: {json.dumps(parsed.get('tool_args', {}), indent=4)}",
            "",
            separator,
            ""
        ])

        full_log = "\n".join(log_lines)

        # Log to console
        logger.info(full_log)

        # Also print to stdout for immediate visibility during testing
        if VERBOSE_RAW_OUTPUT:
            print(full_log)

        # Log to file if enabled
        if LOG_RAW_TO_FILE:
            log_file = os.path.join(RAW_OUTPUT_LOG_DIR, f"session_{self.session_id}.log")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(full_log)
                f.write("\n")

    def _extract_parsed_components(self, text: str) -> Dict[str, Any]:
        """
        Extract all parsed components from the raw response.

        Args:
            text: Raw LLM response

        Returns:
            Dict with thought, tool_name, tool_args, response
        """
        parsed = {}

        # Extract next_thought
        thought_match = re.search(
            r'\[\[ ## next_thought ## \]\](.*?)(?=\[\[|$)',
            text,
            re.DOTALL
        )
        if thought_match:
            parsed['thought'] = thought_match.group(1).strip()

        # Extract next_tool_name
        tool_match = re.search(
            r'\[\[ ## next_tool_name ## \]\](.*?)(?=\[\[|$)',
            text,
            re.DOTALL
        )
        if tool_match:
            parsed['tool_name'] = tool_match.group(1).strip().strip("'\"")

        # Extract next_tool_args
        args_match = re.search(
            r'\[\[ ## next_tool_args ## \]\](.*?)(?=\[\[|$)',
            text,
            re.DOTALL
        )
        if args_match:
            args_str = args_match.group(1).strip()
            try:
                parsed['tool_args'] = json.loads(args_str)
            except json.JSONDecodeError:
                parsed['tool_args'] = args_str

        # Extract response (if present)
        response_match = re.search(
            r'\[\[ ## response ## \]\](.*?)(?=\[\[|$)',
            text,
            re.DOTALL
        )
        if response_match:
            parsed['response'] = response_match.group(1).strip()

        return parsed

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

        # Parse components from raw response
        parsed = self._extract_parsed_components(full_response)

        # Log complete raw output
        self._log_raw_output(
            self.call_count,
            is_thinking,
            prompt,
            messages,
            full_response,
            parsed
        )

        # Capture for training if enabled
        if self.capture_for_training:
            self.captured_outputs.append({
                "call_number": self.call_count,
                "timestamp": datetime.now().isoformat(),
                "is_thinking": is_thinking,
                "prompt": prompt,
                "messages": messages,
                "raw_output": full_response,
                "parsed": parsed
            })

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

    def get_captured_outputs(self) -> List[Dict[str, Any]]:
        """
        Get all captured outputs for training data generation.

        Returns:
            List of captured output dictionaries
        """
        return self.captured_outputs

    def save_captured_outputs(self, filepath: str):
        """
        Save captured outputs to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.captured_outputs, f, indent=2, default=str)
        logger.info(f"Saved {len(self.captured_outputs)} captured outputs to {filepath}")

    def clear_captured_outputs(self):
        """Clear all captured outputs."""
        self.captured_outputs = []
