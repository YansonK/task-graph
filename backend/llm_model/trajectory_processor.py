"""Trajectory processor for handling ReAct agent tool calls and graph updates."""
import logging
from typing import Dict, Any, List
from .utils import parse_tool_data, validate_node_data, is_error_message
from .config import Config

logger = logging.getLogger(__name__)


class TrajectoryProcessor:
    """Processes ReAct agent trajectories and updates the task graph."""

    def __init__(self, max_iters: int = None):
        """
        Initialize the trajectory processor.

        Args:
            max_iters: Maximum iterations to process (defaults to config value)
        """
        self.max_iters = max_iters or Config.MAX_AGENT_ITERATIONS

    def process_trajectory(self, result: Any, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the complete trajectory and update graph data.

        Args:
            result: ReAct agent result with trajectory
            graph_data: Current graph state

        Returns:
            Updated graph data
        """
        for i in range(self.max_iters):
            current_tool = f"tool_name_{i}"
            tool_result = f'observation_{i}'

            # Check if tool exists in trajectory
            if current_tool not in result.trajectory:
                break

            tool_name = result.trajectory[current_tool]

            # Process based on tool type
            if tool_name == "create_task_node":
                self._process_create_node(result.trajectory[tool_result], graph_data)
            elif tool_name == "edit_task_node":
                self._process_edit_node(result.trajectory[tool_result], graph_data)
            else:
                # Unknown tool or finish
                break

        return graph_data

    def _process_create_node(self, node_data: Any, graph_data: Dict[str, Any]) -> None:
        """
        Process a create_task_node tool call.

        Args:
            node_data: Raw node data from tool result
            graph_data: Current graph state (modified in place)
        """
        # Check for error messages
        if is_error_message(node_data):
            logger.error(f"Tool execution failed: {node_data}")
            return

        # Parse node data
        node = parse_tool_data(node_data)
        if node is None:
            return

        # Validate node has required fields
        if not validate_node_data(node, require_description=True):
            return

        # Validate parent exists if parent_id is specified
        parent_id = node.get("parent_id")
        if parent_id:
            parent_exists = self._parent_exists(parent_id, graph_data["nodes"])
            if not parent_exists:
                logger.warning(
                    f"Parent node {parent_id} not found for node {node['id']}, "
                    f"skipping link creation"
                )
                parent_id = None

        # Create link if parent exists
        if parent_id:
            graph_data["links"].append({
                "source": parent_id,
                "target": node["id"]
            })

        # Ensure description is not None
        description = node.get("description", "")

        # Add node to graph
        graph_data["nodes"].append({
            "id": node["id"],
            "name": node["name"],
            "description": description
        })

        logger.info(f"Created task node: {node['name']}")

    def _process_edit_node(self, edit_data: Any, graph_data: Dict[str, Any]) -> None:
        """
        Process an edit_task_node tool call.

        Args:
            edit_data: Raw edit data from tool result
            graph_data: Current graph state (modified in place)
        """
        # Check for error messages
        if is_error_message(edit_data):
            logger.error(f"Tool execution failed: {edit_data}")
            return

        # Parse edit data
        edit_info = parse_tool_data(edit_data)
        if edit_info is None:
            return

        # Validate edit has required fields
        if not isinstance(edit_info, dict) or "id" not in edit_info:
            logger.error(f"Invalid edit data: {edit_info}")
            return

        # Find the node to edit
        node_to_edit = self._find_node_by_id(edit_info["id"], graph_data["nodes"])
        if not node_to_edit:
            logger.error(f"Node not found for editing: {edit_info['id']}")
            return

        # Update node fields if provided (with validation)
        if "name" in edit_info and edit_info["name"]:
            node_to_edit["name"] = edit_info["name"]

        if "description" in edit_info:
            # Allow empty description but not None
            node_to_edit["description"] = (
                edit_info["description"]
                if edit_info["description"] is not None
                else node_to_edit.get("description", "")
            )

        # Handle parent_id change if provided
        if "parent_id" in edit_info:
            self._update_node_parent(
                node_id=edit_info["id"],
                new_parent_id=edit_info["parent_id"],
                graph_data=graph_data
            )

        logger.info(f"Edited task node: {node_to_edit['name']}")

    def _update_node_parent(
        self,
        node_id: str,
        new_parent_id: Any,
        graph_data: Dict[str, Any]
    ) -> None:
        """
        Update a node's parent relationship.

        Args:
            node_id: ID of the node to update
            new_parent_id: New parent ID (or None to remove parent)
            graph_data: Current graph state (modified in place)
        """
        # Remove any existing links where this node is the target
        graph_data["links"] = [
            link for link in graph_data["links"]
            if link["target"] != node_id
        ]

        # Add new parent link if parent_id is not None
        if new_parent_id:
            # Validate parent exists
            if self._parent_exists(new_parent_id, graph_data["nodes"]):
                graph_data["links"].append({
                    "source": new_parent_id,
                    "target": node_id
                })
            else:
                logger.warning(
                    f"Parent node {new_parent_id} not found, skipping link creation"
                )

    @staticmethod
    def _find_node_by_id(node_id: str, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find a node by its ID.

        Args:
            node_id: ID to search for
            nodes: List of node dictionaries

        Returns:
            Node dictionary or None if not found
        """
        for node in nodes:
            if node["id"] == node_id:
                return node
        return None

    @staticmethod
    def _parent_exists(parent_id: str, nodes: List[Dict[str, Any]]) -> bool:
        """
        Check if a parent node exists.

        Args:
            parent_id: Parent ID to check
            nodes: List of node dictionaries

        Returns:
            True if parent exists, False otherwise
        """
        return any(n["id"] == parent_id for n in nodes)
