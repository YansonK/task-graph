"""Utility functions for parsing and validation."""
import json
import ast
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


def parse_tool_data(data: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Parse tool data from string or dict format.

    Tries multiple parsing strategies:
    1. Return as-is if already a dict
    2. JSON parsing
    3. AST literal_eval as fallback

    Args:
        data: Tool data as string or dict

    Returns:
        Parsed dictionary or None if parsing fails
    """
    # Already a dict
    if isinstance(data, dict):
        return data

    # Parse string data
    if isinstance(data, str):
        # Try JSON parsing first
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # Try AST literal_eval as fallback (for dict string representation)
            try:
                return ast.literal_eval(data)
            except (SyntaxError, ValueError) as e:
                logger.error(f"Failed to parse tool data: {data}. Error: {e}")
                return None

    logger.error(f"Unexpected data type: {type(data)}")
    return None


def validate_node_data(node: Dict[str, Any], require_description: bool = False) -> bool:
    """
    Validate that node data has required fields.

    Args:
        node: Node dictionary to validate
        require_description: Whether description field is required

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(node, dict):
        logger.error(f"Node data is not a dict: {type(node)}")
        return False

    required_fields = ["id", "name"]
    if require_description:
        required_fields.append("description")

    for field in required_fields:
        if field not in node:
            logger.error(f"Node missing required field '{field}': {node}")
            return False

    return True


def is_error_message(data: Any) -> bool:
    """
    Check if data represents an error message.

    Args:
        data: Data to check

    Returns:
        True if data appears to be an error message
    """
    if isinstance(data, str):
        data_lower = data.lower()
        return "error" in data_lower or "execution error" in data_lower
    return False


def format_json_safely(data: Any) -> str:
    """
    Safely format data as JSON with fallback to string representation.

    Args:
        data: Data to format

    Returns:
        Formatted string
    """
    try:
        if isinstance(data, str):
            # Try to parse and re-format
            parsed = json.loads(data)
            return json.dumps(parsed, indent=2)
        else:
            return json.dumps(data, indent=2)
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(data)
