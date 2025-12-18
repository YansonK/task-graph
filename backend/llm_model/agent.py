"""
AI Agent for task graph management.

This module provides an agent that:
- Uses DSPy's ReAct pattern for reasoning
- Streams responses in real-time
- Manages task graph operations incrementally
"""

import os
import dspy
import asyncio
import logging
import queue
import threading
from typing import AsyncGenerator, Dict, Any
from openai import AsyncOpenAI

from .tools import tools, TaskBreakdownSignature
from .streaming_lm import StreamingLM
from .graph_operations import GraphOperations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MAX_ITERS = 5


class Agent:
    """
    AI Agent for managing task graphs with streaming capabilities.
    """

    def __init__(self):
        """Initialize the agent with OpenAI configuration."""
        # Get and validate API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        if 'your_openai_api_key_here' in api_key.lower() or 'sk-' not in api_key:
            raise ValueError(
                "OPENAI_API_KEY appears to be invalid. "
                "Please set a real OpenAI API key in backend/.env"
            )

        logger.info(f"Initializing Agent with OpenAI API key: {api_key[:10]}...")

        # Store API key
        self.api_key = api_key

        # Initialize AsyncOpenAI client
        self.openai_client = AsyncOpenAI(api_key=api_key)

        # Configure DSPy with OpenAI GPT-4o-mini
        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', api_key=api_key))

        # Create base ReAct agent
        self.react_agent = dspy.ReAct(
            TaskBreakdownSignature,
            tools=list(tools.values()),
            max_iters=MAX_ITERS
        )

    def _create_streaming_tools(
        self,
        graph_data: Dict[str, Any],
        stream_queue: queue.Queue
    ) -> dict:
        """
        Create wrapped versions of tools that stream updates in real-time.

        Args:
            graph_data: Current graph data (modified in place)
            stream_queue: Queue for streaming updates

        Returns:
            Dictionary of wrapped tools
        """
        from .tools import create_task_node, edit_task_node, finish

        def wrapped_create_task_node(task_name: str, task_description: str, parent_id: str = None):
            """Wrapped create_task_node that streams updates immediately."""
            # Call original tool
            result = create_task_node(task_name, task_description, parent_id)

            # Process and update graph immediately
            graph_update = GraphOperations.create_task_node(result, graph_data)

            # Queue update for streaming
            if graph_update:
                import copy
                stream_queue.put((
                    'graph_update',
                    {
                        'action': graph_update,
                        'graph_data': copy.deepcopy(graph_data)
                    }
                ))

            return result

        def wrapped_edit_task_node(node_id: str, name: str = None, description: str = None, parent_id: str = None):
            """Wrapped edit_task_node that streams updates immediately."""
            # Call original tool
            result = edit_task_node(node_id, name, description, parent_id)

            # Process and update graph immediately
            graph_update = GraphOperations.edit_task_node(result, graph_data)

            # Queue update for streaming
            if graph_update:
                import copy
                stream_queue.put((
                    'graph_update',
                    {
                        'action': graph_update,
                        'graph_data': copy.deepcopy(graph_data)
                    }
                ))

            return result

        # Return wrapped tools as DSPy Tool objects
        return {
            "create_task_node": dspy.Tool(wrapped_create_task_node),
            "edit_task_node": dspy.Tool(wrapped_edit_task_node),
            "finish": dspy.Tool(finish)
        }

    def run_react_agent(
        self,
        chat_history: list,
        graph_data: Dict[str, Any],
        stream_queue: queue.Queue,
        agent_result: dict,
        agent_error: dict,
        agent_done: threading.Event
    ):
        """
        Run the ReAct agent in a background thread with streaming.

        This method:
        1. Creates a streaming LM instance
        2. Creates wrapped tools that stream updates immediately
        3. Runs the ReAct agent with streaming tools
        4. All updates stream in real-time as tools execute

        Args:
            chat_history: Chat message history
            graph_data: Current graph data (modified in place)
            stream_queue: Queue for streaming updates
            agent_result: Dict to store final result
            agent_error: Dict to store any errors
            agent_done: Event to signal completion
        """
        try:
            # Create streaming LM instance
            streaming_lm = StreamingLM(
                model='openai/gpt-4o-mini',
                api_key=self.api_key,
                stream_queue=stream_queue
            )

            # Create wrapped tools that stream updates immediately
            streaming_tools = self._create_streaming_tools(graph_data, stream_queue)

            # Create a custom ReAct agent with streaming tools
            from .tools import TaskBreakdownSignature
            streaming_react = dspy.ReAct(
                TaskBreakdownSignature,
                tools=list(streaming_tools.values()),
                max_iters=MAX_ITERS
            )

            # Run ReAct with streaming LM and tools
            with dspy.context(lm=streaming_lm):
                result = streaming_react(
                    conversation_history=chat_history,
                    task_nodes=graph_data
                )

                agent_result['result'] = result

        except Exception as e:
            agent_error['error'] = e
            logger.error(f"ReAct agent error: {e}")
        finally:
            agent_done.set()

    async def query_stream(
        self,
        chat_history: list,
        graph_data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query responses with real-time updates.

        This provides TRUE streaming where:
        - Thinking tokens stream as they're generated
        - Tool results stream as they execute
        - Graph updates stream incrementally

        Args:
            chat_history: List of chat messages
            graph_data: Current graph data

        Yields:
            Dict with 'type' and corresponding data:
            - {'type': 'token', 'content': str} - Response text chunk
            - {'type': 'thinking', 'content': str} - Reasoning text chunk
            - {'type': 'graph_update', 'graph_data': dict, 'action': dict} - Graph update
        """
        try:
            # Queue to communicate between sync ReAct and async streaming
            stream_queue = queue.Queue()
            agent_done = threading.Event()
            agent_result = {}
            agent_error = {}

            # Start the agent in background thread
            agent_thread = threading.Thread(
                target=self.run_react_agent,
                args=(
                    chat_history,
                    graph_data,
                    stream_queue,
                    agent_result,
                    agent_error,
                    agent_done
                )
            )
            agent_thread.start()

            # Stream updates from queue as they arrive
            while not agent_done.is_set() or not stream_queue.empty():
                try:
                    # Try to get updates from queue
                    msg_type, content = stream_queue.get(timeout=0.02)

                    if msg_type == 'token':
                        yield {'type': 'token', 'content': content}

                    elif msg_type == 'thinking':
                        yield {'type': 'thinking', 'content': content}

                    elif msg_type == 'graph_update':
                        logger.info(f"📤 Sending graph update: {content['action']}")
                        yield {
                            'type': 'graph_update',
                            'graph_data': content['graph_data']
                        }

                except queue.Empty:
                    # No updates available, minimal wait before checking again
                    await asyncio.sleep(0.001)

                elif isinstance(chunk, dspy.Prediction):
                    # Log the completed thought if we accumulated any
                    if current_thought:
                        logger.info(f"💭 Thought: {current_thought}")
                        current_thought = ""

                    # This is the final result with all trajectory information
                    final_result = chunk
                    logger.info(f"✓ Completed reasoning")

            # Send final graph update to ensure client has latest state
            logger.info("📤 Sending final graph update")
            yield {
                'type': 'graph_update',
                'graph_data': graph_data
            }
            logger.info("✅ Stream completed successfully")

            logger.info("✅ Stream completed successfully")

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield {
                'type': 'token',
                'content': f"Sorry, I encountered an error: {str(e)}"
            }

    async def _process_graph_updates(self, result: dspy.Prediction, graph_data: Dict[str, Any]) -> None:
        """
        Process tool calls from ReAct trajectory and update the graph.

        Args:
            result: The final DSPy Prediction with trajectory
            graph_data: The graph data structure to update
        """
        for i in range(max_iters):
            current_tool = f"tool_name_{i}"
            tool_result = f'observation_{i}'

            if current_tool not in result.trajectory:
                break

            if result.trajectory[current_tool] == "create_task_node":
                await self._create_task_node(result.trajectory[tool_result], graph_data)
            elif result.trajectory[current_tool] == "edit_task_node":
                await self._edit_task_node(result.trajectory[tool_result], graph_data)

    async def _create_task_node(self, node_data: Any, graph_data: Dict[str, Any]) -> None:
        """Create a new task node in the graph."""
        # Check if this is an error message
        if isinstance(node_data, str) and ("error" in node_data.lower() or "execution error" in node_data.lower()):
            logger.error(f"Tool execution failed: {node_data}")
            return

        # Parse if it's a string (JSON or dict representation)
        node = self._parse_json_data(node_data, "node")
        if not node:
            return

        # Validate node has required fields
        if not isinstance(node, dict) or "id" not in node or "name" not in node or "description" not in node:
            logger.error(f"Invalid node data: {node}")
            return

        # Validate parent exists if parent_id is specified
        parent_id = node.get("parent_id")
        if parent_id:
            parent_exists = any(n["id"] == parent_id for n in graph_data["nodes"])
            if not parent_exists:
                logger.warning(f"Parent node {parent_id} not found for node {node['id']}, skipping link creation")
                parent_id = None

        # Create link if parent exists
        if parent_id:
            graph_data["links"].append({
                "source": parent_id,
                "target": node["id"]
            })

        # Ensure description is not None
        description = node.get("description", "")

        graph_data["nodes"].append({
            "id": node["id"],
            "name": node["name"],
            "description": description
        })

        logger.info(f"📝 Created node: {node['name']}")

    async def _edit_task_node(self, edit_data: Any, graph_data: Dict[str, Any]) -> None:
        """Edit an existing task node in the graph."""
        # Check if this is an error message
        if isinstance(edit_data, str) and ("error" in edit_data.lower() or "execution error" in edit_data.lower()):
            logger.error(f"Tool execution failed: {edit_data}")
            return

        # Parse if it's a string (JSON or dict representation)
        edit_info = self._parse_json_data(edit_data, "edit")
        if not edit_info:
            return

        # Validate edit has required fields
        if not isinstance(edit_info, dict) or "id" not in edit_info:
            logger.error(f"Invalid edit data: {edit_info}")
            return

        # Find the node to edit
        node_to_edit = None
        for node in graph_data["nodes"]:
            if node["id"] == edit_info["id"]:
                node_to_edit = node
                break

        if not node_to_edit:
            logger.error(f"Node not found for editing: {edit_info['id']}")
            return

        # Update node fields if provided (with validation)
        if "name" in edit_info and edit_info["name"]:
            node_to_edit["name"] = edit_info["name"]
        if "description" in edit_info:
            # Allow empty description but not None
            node_to_edit["description"] = edit_info["description"] if edit_info["description"] is not None else node_to_edit.get("description", "")

        # Handle parent_id change if provided
        if "parent_id" in edit_info:
            # Remove any existing links where this node is the target
            graph_data["links"] = [
                link for link in graph_data["links"]
                if link["target"] != edit_info["id"]
            ]

            # Add new parent link if parent_id is not None
            new_parent_id = edit_info["parent_id"]
            if new_parent_id:
                # Validate parent exists
                parent_exists = any(n["id"] == new_parent_id for n in graph_data["nodes"])
                if parent_exists:
                    graph_data["links"].append({
                        "source": new_parent_id,
                        "target": edit_info["id"]
                    })
                else:
                    logger.warning(f"Parent node {new_parent_id} not found, skipping link creation")

        logger.info(f"✏️  Edited node: {node_to_edit['name']}")

    def _parse_json_data(self, data: Any, data_type: str) -> Any:
        """
        Parse JSON data that might be a string or already parsed.

        Args:
            data: The data to parse (string or dict)
            data_type: Description of data type for logging

        Returns:
            Parsed data or None if parsing fails
        """
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Try eval as fallback (for dict string representation)
                try:
                    import ast
                    return ast.literal_eval(data)
                except (SyntaxError, ValueError) as e:
                    logger.error(f"Failed to parse {data_type} data: {data}. Error: {e}")
                    return None
        else:
            return data

    def query(self, chat_history, graph_data):
        """
        Non-streaming query method for backward compatibility.

        Args:
            chat_history: List of conversation messages
            graph_data: Current graph data structure

        Returns:
            DSPy Prediction with response and trajectory
        """
        return self.react_agent(
            conversation_history=chat_history,
            task_nodes=graph_data
        )
