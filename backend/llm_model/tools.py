import dspy
from dataclasses import dataclass
from typing import Optional
import json
import datetime

def create_task_node(task_name: str, task_description: str, parent_id: Optional[str] = None) -> dict:
    """Create a new task node with an optional parent."""
    # Generate a unique node ID (e.g., using a timestamp or UUID)
    node_id = f"node_{datetime.datetime.now().timestamp()}"

    # Return the new node data
    return {
        "id": node_id,
        "name": task_name,
        "description": task_description,
        "parent_id": parent_id
    }

def edit_task_node(node_id: str, name: Optional[str] = None, description: Optional[str] = None, parent_id: Optional[str] = None) -> dict:
    """Edit an existing task node's name, description, or parent_id.

    Args:
        node_id: The ID of the node to edit (required)
        name: New name for the node (optional)
        description: New description for the node (optional)
        parent_id: New parent ID for the node (optional, use "null" to remove parent)

    Returns:
        Dictionary with the node_id and updated fields
    """
    result = {"id": node_id, "edit": True}

    if name is not None:
        result["name"] = name
    if description is not None:
        result["description"] = description
    if parent_id is not None:
        # Allow "null" string to explicitly remove parent
        result["parent_id"] = None if parent_id == "null" else parent_id

    return result

def update_task_status(node_id: str, status: str) -> dict:
    """Update the status of an existing task node.

    Args:
        node_id: The ID of the node to update (required)
        status: New status for the node. Must be one of: 'notStarted', 'inProgress', 'completed'

    Returns:
        Dictionary with the node_id and new status
    """
    return {
        "id": node_id,
        "status": status
    }

def delete_task_node(node_id: str) -> dict:
    """Delete a task node from the graph.

    When a node is deleted:
    - If the node has a parent: all children are reconnected to the parent
    - If the node has no parent (root): all descendants are cascade deleted

    Args:
        node_id: The ID of the node to delete (required)

    Returns:
        Dictionary with the node_id to delete
    """
    return {
        "id": node_id
    }

def finish():
    """Conclude the trajectory."""
    return "Finish"

# Tools dictionary
tools = {
    "create_task_node": dspy.Tool(create_task_node),
    "edit_task_node": dspy.Tool(edit_task_node),
    "update_task_status": dspy.Tool(update_task_status),
    "delete_task_node": dspy.Tool(delete_task_node),
    "finish": dspy.Tool(finish)  # To allow the agent to finish
}

class TaskBreakdownSignature(dspy.Signature):
    """Agent for breaking down tasks and managing a task graph. Will guide users to more specific tasks before commiting to creating new task nodes.

    You can track the progress of tasks by updating their status:
    - 'notStarted': Task has been created but not yet started (default for new tasks)
    - 'inProgress': Task is currently being worked on
    - 'completed': Task has been finished

    Use the update_task_status tool to transition tasks through these states as work progresses. When you begin working on a task, mark it as 'inProgress'. When you finish a task, mark it as 'completed'."""
    conversation_history: str = dspy.InputField()
    task_nodes: dict = dspy.InputField(desc="Current list of task nodes and their links. Each node has an optional 'status' field that can be 'notStarted', 'inProgress', or 'completed'")
    response: str = dspy.OutputField(desc="Agent's reply to the user")
