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
logging.basicConfig(level=logging.INFO)
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
        from .tools import create_task_node, edit_task_node, update_task_status, finish

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

        def wrapped_update_task_status(node_id: str, status: str):
            """Wrapped update_task_status that streams updates immediately."""
            # Call original tool
            result = update_task_status(node_id, status)

            # Process and update graph immediately
            graph_update = GraphOperations.update_task_status(result, graph_data)

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
            "update_task_status": dspy.Tool(wrapped_update_task_status),
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

            # Wait for thread to complete
            agent_thread.join()

            # Check for errors
            if 'error' in agent_error:
                raise agent_error['error']

            # Send final graph update to ensure client has latest state
            logger.info("📤 Sending final graph update")
            yield {
                'type': 'graph_update',
                'graph_data': graph_data
            }

            logger.info("✅ Stream completed successfully")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                'type': 'token',
                'content': f"Sorry, I encountered an error: {str(e)}"
            }
