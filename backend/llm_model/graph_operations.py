"""
Graph operations module for managing task nodes and links.

This module handles all graph manipulation operations including:
- Creating task nodes
- Editing task nodes
- Managing links between nodes
- Validating graph operations
"""

import json
import logging
from typing import Dict, Any, Optional, List
import ast

logger = logging.getLogger(__name__)


class GraphOperations:
    """Handles all graph manipulation operations."""

    @staticmethod
    def parse_tool_result(tool_result: Any) -> Optional[Dict[str, Any]]:
        """
        Parse tool result from various formats (string, dict, JSON).

        Args:
            tool_result: The result from a tool execution

        Returns:
            Parsed dictionary or None if parsing fails
        """
        # Check if this is an error message
        if isinstance(tool_result, str) and (
            "error" in tool_result.lower() or
            "execution error" in tool_result.lower()
        ):
            logger.error(f"Tool execution failed: {tool_result}")
            return None

        # Parse if it's a string (JSON or dict representation)
        if isinstance(tool_result, str):
            try:
                return json.loads(tool_result)
            except json.JSONDecodeError:
                # Try eval as fallback (for dict string representation)
                try:
                    return ast.literal_eval(tool_result)
                except (SyntaxError, ValueError) as e:
                    logger.error(f"Failed to parse tool result: {tool_result}. Error: {e}")
                    return None

        return tool_result

    @staticmethod
    def validate_node_data(node: Dict[str, Any]) -> bool:
        """
        Validate that node data has required fields.

        Args:
            node: Node data dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(node, dict):
            logger.error(f"Node data is not a dictionary: {node}")
            return False

        required_fields = ["id", "name", "description"]
        for field in required_fields:
            if field not in node:
                logger.error(f"Node missing required field '{field}': {node}")
                return False

        return True

    @staticmethod
    def parent_exists(parent_id: str, graph_data: Dict[str, Any]) -> bool:
        """
        Check if a parent node exists in the graph.

        Args:
            parent_id: ID of the parent node
            graph_data: Current graph data

        Returns:
            True if parent exists, False otherwise
        """
        return any(n["id"] == parent_id for n in graph_data["nodes"])

    @staticmethod
    def create_task_node(
        node_data: Any,
        graph_data: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Create a new task node in the graph.

        Args:
            node_data: Node data (string or dict)
            graph_data: Current graph data (modified in place)

        Returns:
            Dictionary with node info or None if creation failed
        """
        # Parse node data
        node = GraphOperations.parse_tool_result(node_data)
        if node is None:
            return None

        # Validate node data
        if not GraphOperations.validate_node_data(node):
            return None

        # Validate parent exists if parent_id is specified
        parent_id = node.get("parent_id")
        if parent_id:
            if not GraphOperations.parent_exists(parent_id, graph_data):
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
            "description": description,
            "status": "notStarted"  # All new nodes start as not started
        })

        logger.info(f"📝 Created node: {node['name']}")

        return {
            "action": "create",
            "name": node["name"],
            "id": node["id"]
        }

    @staticmethod
    def edit_task_node(
        edit_data: Any,
        graph_data: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Edit an existing task node in the graph.

        Args:
            edit_data: Edit data (string or dict)
            graph_data: Current graph data (modified in place)

        Returns:
            Dictionary with edit info or None if edit failed
        """
        # Parse edit data
        edit_info = GraphOperations.parse_tool_result(edit_data)
        if edit_info is None:
            return None

        # Validate edit has required fields
        if not isinstance(edit_info, dict) or "id" not in edit_info:
            logger.error(f"Invalid edit data: {edit_info}")
            return None

        # Find the node to edit
        node_to_edit = None
        for node in graph_data["nodes"]:
            if node["id"] == edit_info["id"]:
                node_to_edit = node
                break

        if not node_to_edit:
            logger.error(f"Node not found for editing: {edit_info['id']}")
            return None

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
            # Remove any existing links where this node is the target
            graph_data["links"] = [
                link for link in graph_data["links"]
                if link["target"] != edit_info["id"]
            ]

            # Add new parent link if parent_id is not None
            new_parent_id = edit_info["parent_id"]
            if new_parent_id:
                # Validate parent exists
                if GraphOperations.parent_exists(new_parent_id, graph_data):
                    graph_data["links"].append({
                        "source": new_parent_id,
                        "target": edit_info["id"]
                    })
                else:
                    logger.warning(
                        f"Parent node {new_parent_id} not found, "
                        f"skipping link creation"
                    )

        logger.info(f"✏️  Edited node: {node_to_edit['name']}")

        return {
            "action": "edit",
            "name": node_to_edit["name"],
            "id": edit_info["id"]
        }

    @staticmethod
    def update_task_status(
        status_data: Any,
        graph_data: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Update the status of an existing task node.

        Args:
            status_data: Status update data (string or dict) with 'id' and 'status'
            graph_data: Current graph data (modified in place)

        Returns:
            Dictionary with update info or None if update failed
        """
        # Parse status data
        update_info = GraphOperations.parse_tool_result(status_data)
        if update_info is None:
            return None

        # Validate update has required fields
        if not isinstance(update_info, dict) or "id" not in update_info or "status" not in update_info:
            logger.error(f"Invalid status update data: {update_info}")
            return None

        # Validate status value
        valid_statuses = ["notStarted", "inProgress", "completed"]
        new_status = update_info["status"]
        if new_status not in valid_statuses:
            logger.error(
                f"Invalid status value '{new_status}'. "
                f"Must be one of: {', '.join(valid_statuses)}"
            )
            return None

        # Find the node to update
        node_to_update = None
        for node in graph_data["nodes"]:
            if node["id"] == update_info["id"]:
                node_to_update = node
                break

        if not node_to_update:
            logger.error(f"Node not found for status update: {update_info['id']}")
            return None

        # Update node status
        old_status = node_to_update.get("status", "notStarted")
        node_to_update["status"] = new_status

        logger.info(
            f"🔄 Updated status of '{node_to_update['name']}': "
            f"{old_status} → {new_status}"
        )

        return {
            "action": "update_status",
            "name": node_to_update["name"],
            "id": update_info["id"],
            "old_status": old_status,
            "new_status": new_status
        }

    @staticmethod
    def delete_task_node(
        delete_data: Any,
        graph_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Delete a task node from the graph.

        When a node is deleted:
        - If the node has a parent: all children are reconnected to the parent
        - If the node has no parent (root): all descendants are cascade deleted

        Args:
            delete_data: Delete data (string or dict) with 'id'
            graph_data: Current graph data (modified in place)

        Returns:
            Dictionary with deletion info or None if deletion failed
        """
        # Parse delete data
        delete_info = GraphOperations.parse_tool_result(delete_data)
        if delete_info is None:
            return None

        # Validate delete has required fields
        if not isinstance(delete_info, dict) or "id" not in delete_info:
            logger.error(f"Invalid delete data: {delete_info}")
            return None

        node_id = delete_info["id"]

        # Find the node to delete
        node_to_delete = None
        for node in graph_data["nodes"]:
            if node["id"] == node_id:
                node_to_delete = node
                break

        if not node_to_delete:
            logger.error(f"Node not found for deletion: {node_id}")
            return None

        # Find parent of the node to delete
        parent_id = None
        for link in graph_data["links"]:
            if link["target"] == node_id:
                parent_id = link["source"]
                break

        # Find all children of the node to delete
        children_ids = [
            link["target"] for link in graph_data["links"]
            if link["source"] == node_id
        ]

        deleted_nodes = []

        if parent_id:
            # Case 1: Node has a parent - reconnect children to parent
            # Remove all links involving this node
            graph_data["links"] = [
                link for link in graph_data["links"]
                if link["source"] != node_id and link["target"] != node_id
            ]

            # Reconnect children to parent
            for child_id in children_ids:
                graph_data["links"].append({
                    "source": parent_id,
                    "target": child_id
                })

            # Remove the node
            graph_data["nodes"] = [
                node for node in graph_data["nodes"]
                if node["id"] != node_id
            ]

            deleted_nodes.append(node_id)

            logger.info(
                f"🗑️  Deleted node: {node_to_delete['name']} "
                f"(reconnected {len(children_ids)} children to parent)"
            )
        else:
            # Case 2: Node has no parent (root) - cascade delete all descendants
            def find_all_descendants(nid: str, visited: set = None) -> List[str]:
                """Recursively find all descendant node IDs."""
                if visited is None:
                    visited = set()

                if nid in visited:
                    return []

                visited.add(nid)
                descendants = [nid]

                # Find direct children
                direct_children = [
                    link["target"] for link in graph_data["links"]
                    if link["source"] == nid
                ]

                # Recursively find descendants of children
                for child_id in direct_children:
                    descendants.extend(find_all_descendants(child_id, visited))

                return descendants

            # Find all nodes to delete (node + all descendants)
            nodes_to_delete = find_all_descendants(node_id)
            deleted_nodes.extend(nodes_to_delete)

            # Remove all links involving any of these nodes
            graph_data["links"] = [
                link for link in graph_data["links"]
                if link["source"] not in nodes_to_delete and link["target"] not in nodes_to_delete
            ]

            # Remove all the nodes
            graph_data["nodes"] = [
                node for node in graph_data["nodes"]
                if node["id"] not in nodes_to_delete
            ]

            logger.info(
                f"🗑️  Cascade deleted root node: {node_to_delete['name']} "
                f"(deleted {len(nodes_to_delete)} total nodes)"
            )

        return {
            "action": "delete",
            "name": node_to_delete["name"],
            "id": node_id,
            "deleted_nodes": deleted_nodes,
            "reconnected_children": len(children_ids) if parent_id else 0,
            "cascade_deleted": len(deleted_nodes) if not parent_id else 0
        }
