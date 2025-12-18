# Fine-Tuning Examples Outline for Task Graph Agent

## Overview

This document outlines the comprehensive set of training examples needed to fine-tune the task graph agent. The examples are designed to cover all tool usage patterns, conversation scenarios, and edge cases that the agent should handle correctly.

## DSPy Training Example Format

### Basic Structure
```python
import dspy

# Each example follows this format
example = dspy.Example(
    conversation_history="User: <user message>\nAssistant: <optional prior response>",
    task_nodes={
        "nodes": [...],
        "links": [...]
    },
    response="<expected agent response>"
).with_inputs("conversation_history", "task_nodes")
```

### Key Points for Training Data
- **conversation_history**: String containing the chat history in "User: ... Assistant: ..." format
- **task_nodes**: Dict containing current graph state with `nodes` and `links` arrays
- **response**: The ideal response the agent should produce (for labeled examples)
- Use `.with_inputs()` to mark which fields are inputs vs. labels

---

## ReAct Agent Trajectory Format (Critical for Tool-Using Agents)

When training a ReAct agent with tool access, the training examples must include the **trajectory** - the complete reasoning and tool-calling trace. This is how the agent learns to interleave thinking and tool usage.

### Trajectory Structure

The trajectory is a dictionary that accumulates step-by-step agent reasoning:

```python
trajectory = {
    # Step 0
    "thought_0": "I need to break down this project into subtasks...",
    "tool_name_0": "create_task_node",
    "tool_args_0": {"task_name": "Web App", "task_description": "Full-stack web application"},
    "observation_0": {"id": "node_1", "name": "Web App", "description": "Full-stack web application"},

    # Step 1
    "thought_1": "Now I should add the frontend subtask...",
    "tool_name_1": "create_task_node",
    "tool_args_1": {"task_name": "Frontend", "task_description": "React UI", "parent_id": "node_1"},
    "observation_1": {"id": "node_2", "name": "Frontend", "description": "React UI", "parent_id": "node_1"},

    # Step 2 (finishing)
    "thought_2": "I've created the task breakdown. Time to finish.",
    "tool_name_2": "finish",
    "tool_args_2": {},
    "observation_2": "Completed."
}
```

### ReAct Output Fields

Each step in the ReAct loop produces three outputs:

| Field | Type | Description |
|-------|------|-------------|
| `next_thought` | `str` | Agent's reasoning about the current situation |
| `next_tool_name` | `Literal[tool_names]` | Selected tool from available options |
| `next_tool_args` | `dict[str, Any]` | Arguments for the tool call |

### Complete Training Example with Trajectory

```python
import dspy
from typing import Literal

# Full training example for a ReAct agent
training_example = dspy.Example(
    # Input fields
    conversation_history="User: I want to build a blog platform with authentication",
    task_nodes={"nodes": [], "links": []},

    # The trajectory captures the full reasoning + tool use sequence
    trajectory={
        # Step 0: Create main task
        "thought_0": "The user wants to build a blog platform with authentication. I'll create the main project task first.",
        "tool_name_0": "create_task_node",
        "tool_args_0": {
            "task_name": "Blog Platform",
            "task_description": "Full-stack blog with user authentication"
        },
        "observation_0": {
            "id": "node_1",
            "name": "Blog Platform",
            "description": "Full-stack blog with user authentication"
        },

        # Step 1: Create authentication subtask
        "thought_1": "Now I need to add the authentication feature as a subtask.",
        "tool_name_1": "create_task_node",
        "tool_args_1": {
            "task_name": "User Authentication",
            "task_description": "Login, registration, password reset, sessions",
            "parent_id": "node_1"
        },
        "observation_1": {
            "id": "node_2",
            "name": "User Authentication",
            "description": "Login, registration, password reset, sessions",
            "parent_id": "node_1"
        },

        # Step 2: Create blog posts subtask
        "thought_2": "I should also add the core blog functionality.",
        "tool_name_2": "create_task_node",
        "tool_args_2": {
            "task_name": "Blog Posts",
            "task_description": "Create, edit, delete, and display blog posts",
            "parent_id": "node_1"
        },
        "observation_2": {
            "id": "node_3",
            "name": "Blog Posts",
            "description": "Create, edit, delete, and display blog posts",
            "parent_id": "node_1"
        },

        # Step 3: Finish
        "thought_3": "I've created a good initial breakdown. The user can expand these further.",
        "tool_name_3": "finish",
        "tool_args_3": {},
        "observation_3": "Completed."
    },

    # Final response to user
    response="I've created a task breakdown for your blog platform with two main subtasks: User Authentication and Blog Posts. Would you like me to break these down further?"

).with_inputs("conversation_history", "task_nodes")
```

### How BootstrapFinetune Uses Trajectories

When using `dspy.BootstrapFinetune`:

1. **Teacher Program Execution**: The teacher (optimized) program runs on training inputs
2. **Trace Collection**: DSPy records the full trajectory including:
   - Each `next_thought` reasoning step
   - Each `next_tool_name` selection
   - Each `next_tool_args` dictionary
   - Each `observation` from tool execution
3. **Training Data Generation**: These traces become prompt-completion pairs
4. **Fine-tuning**: The student model learns to reproduce these reasoning patterns

```python
# Example optimization with trajectory learning
dspy.settings.experimental = True

optimizer = dspy.BootstrapFinetune(
    metric=task_completion_metric,  # Your evaluation metric
    num_threads=8
)

# Teacher is a prompt-optimized ReAct agent
finetuned_agent = optimizer.compile(
    student=student_react_agent,
    teacher=teacher_react_agent,
    trainset=training_examples
)
```

### Trajectory Examples by Scenario

#### Single Tool Call Trajectory
```python
trajectory = {
    "thought_0": "User wants to mark the authentication task as complete.",
    "tool_name_0": "update_task_status",
    "tool_args_0": {"node_id": "node_2", "status": "completed"},
    "observation_0": {"id": "node_2", "status": "completed"},
    "thought_1": "Done. I'll inform the user.",
    "tool_name_1": "finish",
    "tool_args_1": {},
    "observation_1": "Completed."
}
```

#### No Tool Call Trajectory (Clarification)
```python
trajectory = {
    "thought_0": "The user's request is too vague. I need more details before creating tasks.",
    "tool_name_0": "finish",
    "tool_args_0": {},
    "observation_0": "Completed."
}
# response = "Could you tell me more about what you're building? What's the main goal?"
```

#### Multi-Tool Sequence Trajectory
```python
trajectory = {
    "thought_0": "User wants to reorganize - move testing under QA and rename it.",
    "tool_name_0": "edit_task_node",
    "tool_args_0": {"node_id": "node_5", "parent_id": "node_qa", "name": "Quality Assurance Testing"},
    "observation_0": {"id": "node_5", "name": "Quality Assurance Testing", "parent_id": "node_qa"},
    "thought_1": "Successfully moved and renamed. Task complete.",
    "tool_name_1": "finish",
    "tool_args_1": {},
    "observation_1": "Completed."
}
```

---

## Category 1: Single Tool Usage Examples

### 1.1 Create Task Node Examples

**Purpose**: Teach the agent when and how to create new task nodes.

#### 1.1.1 Creating a Root Node (No Parent)
- **Scenario**: User requests a new top-level project/task
- **Expected behavior**: Use `create_task_node` without `parent_id`
- **Example cases**:
  - "I want to build a web application"
  - "Start a new project for inventory management"
  - "Create a task for redesigning our homepage"

#### 1.1.2 Creating a Child Node (With Parent)
- **Scenario**: User wants to add a subtask to an existing node
- **Expected behavior**: Use `create_task_node` with correct `parent_id`
- **Example cases**:
  - "Add a subtask for database design under the backend task"
  - "Break down the authentication task into smaller steps"
  - "Create a testing subtask for the login feature"

#### 1.1.3 Creating Multiple Related Nodes
- **Scenario**: User wants to create several related tasks at once
- **Expected behavior**: Multiple `create_task_node` calls with appropriate hierarchy
- **Example cases**:
  - "Break this into frontend, backend, and database tasks"
  - "Add unit tests, integration tests, and e2e tests as subtasks"

### 1.2 Edit Task Node Examples

**Purpose**: Teach the agent when and how to modify existing nodes.

#### 1.2.1 Editing Node Name Only
- **Scenario**: User wants to rename a task
- **Expected behavior**: Use `edit_task_node` with `name` parameter only
- **Example cases**:
  - "Rename 'DB Setup' to 'Database Configuration'"
  - "Change the name of that task to something more descriptive"

#### 1.2.2 Editing Node Description Only
- **Scenario**: User wants to update task details
- **Expected behavior**: Use `edit_task_node` with `description` parameter only
- **Example cases**:
  - "Add more details to the API task description"
  - "Update the description to include the deadline"

#### 1.2.3 Changing Node Parent (Restructuring)
- **Scenario**: User wants to move a task to a different parent
- **Expected behavior**: Use `edit_task_node` with `parent_id` parameter
- **Example cases**:
  - "Move the testing task under QA instead of development"
  - "Reorganize: put all UI tasks under the frontend node"

#### 1.2.4 Editing Multiple Properties
- **Scenario**: User wants to update several properties at once
- **Expected behavior**: Single `edit_task_node` call with multiple parameters
- **Example cases**:
  - "Update the task name and description together"

### 1.3 Update Task Status Examples

**Purpose**: Teach the agent to correctly manage task status transitions.

#### 1.3.1 Starting a Task (notStarted → inProgress)
- **Scenario**: User indicates they're beginning work
- **Expected behavior**: Use `update_task_status` with status="inProgress"
- **Example cases**:
  - "I'm starting work on the authentication feature"
  - "Mark the database task as in progress"
  - "Begin the frontend development"

#### 1.3.2 Completing a Task (inProgress → completed)
- **Scenario**: User indicates task is finished
- **Expected behavior**: Use `update_task_status` with status="completed"
- **Example cases**:
  - "I've finished the login page"
  - "Mark authentication as complete"
  - "Done with the API endpoints"

#### 1.3.3 Direct Completion (notStarted → completed)
- **Scenario**: Task was completed without being marked as in progress first
- **Expected behavior**: Use `update_task_status` with status="completed"
- **Example cases**:
  - "Actually, I already completed that task"
  - "That one is already done"

#### 1.3.4 Reopening a Task (completed → inProgress/notStarted)
- **Scenario**: A completed task needs more work
- **Expected behavior**: Use `update_task_status` to revert status
- **Example cases**:
  - "Actually, we need to revisit the authentication task"
  - "Reopen the database task - found a bug"

### 1.4 Delete Task Node Examples

**Purpose**: Teach the agent when and how to remove tasks.

#### 1.4.1 Deleting a Leaf Node (No Children)
- **Scenario**: Remove a task with no subtasks
- **Expected behavior**: Simple `delete_task_node` call
- **Example cases**:
  - "Remove the deprecated task"
  - "Delete the 'Setup CI' task - we don't need it anymore"

#### 1.4.2 Deleting a Node with Children (Middle Node)
- **Scenario**: Remove a task that has subtasks, and the task has a parent
- **Expected behavior**: `delete_task_node` - children reconnect to grandparent
- **Example cases**:
  - "Remove the 'Backend' container task but keep its subtasks"
  - "Flatten the hierarchy - remove this intermediate grouping"

#### 1.4.3 Deleting a Root Node with Children
- **Scenario**: Remove a root task that has descendants
- **Expected behavior**: `delete_task_node` - cascades to delete all descendants
- **Example cases**:
  - "Delete the entire project and all its tasks"
  - "Remove everything under 'Legacy System'"

---

## Category 2: Multi-Tool Sequence Examples

### 2.1 Task Breakdown Sequences

**Purpose**: Teach the agent to break down complex tasks into hierarchical subtasks.

#### 2.1.1 Two-Level Breakdown
- **Scenario**: User provides a task, agent creates parent and children
- **Expected behavior**:
  1. `create_task_node` for main task
  2. Multiple `create_task_node` calls for subtasks with parent_id
- **Example cases**:
  - "Plan a website redesign project"
  - "Break down building a REST API"

#### 2.1.2 Three-Level Breakdown
- **Scenario**: Complex task requiring deeper hierarchy
- **Expected behavior**: Create nodes at three levels with proper parent relationships
- **Example cases**:
  - "Plan an e-commerce platform with detailed subtasks"
  - "Create a comprehensive mobile app development plan"

#### 2.1.3 Progressive Breakdown (User-Guided)
- **Scenario**: Agent creates initial breakdown, user asks for more detail on specific areas
- **Expected behavior**: Create additional child nodes under specified parent
- **Example cases**:
  - First: "Break down the project" → Then: "Now break down the backend part further"

### 2.2 Create and Configure Sequences

#### 2.2.1 Create Then Start
- **Scenario**: User creates a task and immediately wants to start it
- **Expected behavior**:
  1. `create_task_node`
  2. `update_task_status` to "inProgress"
- **Example cases**:
  - "Create a task for the homepage and mark it as started"

#### 2.2.2 Create Then Edit
- **Scenario**: User creates a task then wants to modify it
- **Expected behavior**:
  1. `create_task_node`
  2. `edit_task_node` to modify
- **Example cases**:
  - "Add a task... actually, let me rephrase that description"

### 2.3 Reorganization Sequences

#### 2.3.1 Move and Update
- **Scenario**: Restructure task hierarchy and update properties
- **Expected behavior**: Multiple `edit_task_node` calls
- **Example cases**:
  - "Reorganize the development tasks - move testing under QA and rename it"

#### 2.3.2 Delete and Recreate
- **Scenario**: User wants to completely redo a section
- **Expected behavior**:
  1. `delete_task_node`
  2. Multiple `create_task_node` calls
- **Example cases**:
  - "Remove all the backend tasks and create a new structure"

### 2.4 Progress Tracking Sequences

#### 2.4.1 Complete Parent After Children
- **Scenario**: All subtasks completed, mark parent as complete
- **Expected behavior**:
  1. `update_task_status` for last child to "completed"
  2. `update_task_status` for parent to "completed"
- **Example cases**:
  - "Mark testing as done, and since all dev tasks are complete, mark development as finished too"

#### 2.4.2 Batch Status Updates
- **Scenario**: Update status on multiple related tasks
- **Expected behavior**: Multiple `update_task_status` calls
- **Example cases**:
  - "Mark all the UI tasks as in progress"
  - "Complete all the planning subtasks"

---

## Category 3: Conversational Examples (No Tool Use)

### 3.1 Clarification Seeking

**Purpose**: Teach the agent to ask for clarification before creating tasks.

#### 3.1.1 Vague Initial Request
- **Scenario**: User gives very vague task description
- **Expected behavior**: Ask clarifying questions, DON'T create nodes yet
- **Example cases**:
  - User: "I need to build something for my business"
  - User: "Help me with my project"
  - User: "Create some tasks"

#### 3.1.2 Ambiguous Scope
- **Scenario**: Task scope is unclear
- **Expected behavior**: Ask about scope/boundaries
- **Example cases**:
  - User: "Build an app" (What platform? What features?)
  - User: "Set up the infrastructure" (What services? What scale?)

#### 3.1.3 Missing Context
- **Scenario**: Need more information about requirements
- **Expected behavior**: Ask about tech stack, constraints, timeline
- **Example cases**:
  - "Create a backend" (What language? What database?)
  - "Build authentication" (OAuth? JWT? Social login?)

### 3.2 Explanation and Guidance

#### 3.2.1 Explaining the Graph
- **Scenario**: User asks about current task structure
- **Expected behavior**: Describe the graph without modifying it
- **Example cases**:
  - "What tasks do I have?"
  - "Explain the current project structure"
  - "What's the status of everything?"

#### 3.2.2 Suggesting Next Steps
- **Scenario**: User asks what to do next
- **Expected behavior**: Analyze graph and suggest logical next actions
- **Example cases**:
  - "What should I work on next?"
  - "Which task should I prioritize?"
  - "What's blocking progress?"

#### 3.2.3 Best Practices Guidance
- **Scenario**: User asks for advice on task organization
- **Expected behavior**: Provide guidance without modifying graph
- **Example cases**:
  - "How should I structure this project?"
  - "Is this a good task breakdown?"
  - "Should I add more subtasks?"

### 3.3 Confirmation Before Action

#### 3.3.1 Confirming Destructive Actions
- **Scenario**: User asks to delete or major restructure
- **Expected behavior**: Confirm before executing
- **Example cases**:
  - "Delete all the tasks" → Confirm first
  - "Start over completely" → Confirm the scope

#### 3.3.2 Confirming Ambiguous Requests
- **Scenario**: Request could be interpreted multiple ways
- **Expected behavior**: Clarify intent before acting
- **Example cases**:
  - "Remove the backend task" (just that node or all children too?)
  - "Mark it as done" (which task is "it"?)

---

## Category 4: Context-Aware Examples

### 4.1 Graph State Awareness

#### 4.1.1 Empty Graph Handling
- **Scenario**: Starting with no nodes
- **Graph State**: `{"nodes": [], "links": []}`
- **Expected behavior**: Create root node(s) from scratch
- **Example cases**:
  - First interaction on new project
  - After complete graph deletion

#### 4.1.2 Single Node Graph
- **Scenario**: Graph has one main task
- **Graph State**: One node, no links
- **Expected behavior**: Either break down or ask for clarification
- **Example cases**:
  - "Break this down" with single node present

#### 4.1.3 Complex Existing Graph
- **Scenario**: Graph already has substantial structure
- **Graph State**: Multiple nodes and links
- **Expected behavior**: Understand context before modifications
- **Example cases**:
  - Adding to an established project
  - Making targeted modifications

### 4.2 Status-Aware Examples

#### 4.2.1 Working with In-Progress Tasks
- **Scenario**: User references or wants to modify active tasks
- **Graph State**: Some nodes have status="inProgress"
- **Expected behavior**: Respect ongoing work context
- **Example cases**:
  - "How's the current work going?" (reference inProgress tasks)
  - "Finish what I'm working on" (complete inProgress tasks)

#### 4.2.2 Handling Completed Tasks
- **Scenario**: User wants to modify or reference completed work
- **Graph State**: Some nodes have status="completed"
- **Expected behavior**: Handle appropriately, warn about reopening
- **Example cases**:
  - "Add details to the completed task"
  - "Can we mark that as incomplete again?"

#### 4.2.3 Mixed Status Scenarios
- **Scenario**: Graph has nodes in various states
- **Graph State**: Mix of notStarted, inProgress, completed
- **Expected behavior**: Give status-aware responses
- **Example cases**:
  - "What's left to do?" (filter to notStarted)
  - "What have I accomplished?" (filter to completed)

### 4.3 Hierarchy-Aware Examples

#### 4.3.1 Reference by Position
- **Scenario**: User references tasks by their position in hierarchy
- **Expected behavior**: Correctly identify nodes by context
- **Example cases**:
  - "The first subtask of backend"
  - "The parent task"
  - "All children of development"

#### 4.3.2 Sibling Relationships
- **Scenario**: User references related tasks at same level
- **Expected behavior**: Understand sibling context
- **Example cases**:
  - "Add another task like this one"
  - "Apply the same status to all siblings"

---

## Category 5: Error Handling and Edge Cases

### 5.1 Invalid Requests

#### 5.1.1 Non-Existent Node References
- **Scenario**: User references a node that doesn't exist
- **Expected behavior**: Politely indicate the node wasn't found
- **Example cases**:
  - "Update the deployment task" (when no such task exists)
  - "Delete task xyz" (when xyz doesn't exist)

#### 5.1.2 Invalid Status Transitions
- **Scenario**: User requests an invalid state
- **Expected behavior**: Explain valid options
- **Example cases**:
  - "Mark it as 'done'" (should be 'completed')
  - "Set status to 'blocked'" (not a valid status)

#### 5.1.3 Circular Dependency Attempts
- **Scenario**: User tries to create invalid hierarchy
- **Expected behavior**: Prevent and explain
- **Example cases**:
  - Trying to make a node its own parent
  - Trying to create cycles in the graph

### 5.2 Ambiguous References

#### 5.2.1 Multiple Matching Nodes
- **Scenario**: User's description matches multiple nodes
- **Expected behavior**: Ask for clarification
- **Example cases**:
  - "Complete the testing task" (when multiple testing tasks exist)
  - "Edit the API endpoint" (when multiple API tasks exist)

#### 5.2.2 Partial Name Matches
- **Scenario**: User uses partial or similar names
- **Expected behavior**: Identify likely match and confirm
- **Example cases**:
  - "Update auth" (matches "Authentication" node)
  - "The DB task" (matches "Database Setup" node)

### 5.3 Conversation Context

#### 5.3.1 Follow-up Without Context
- **Scenario**: User continues conversation but reference is unclear
- **Expected behavior**: Ask for clarification or use recent context
- **Example cases**:
  - "Now add subtasks to it" (unclear what "it" refers to)
  - "Do the same for the other one"

#### 5.3.2 Contradiction Handling
- **Scenario**: User's request contradicts previous statements
- **Expected behavior**: Clarify the contradiction
- **Example cases**:
  - "Add that task again" (after just deleting it)
  - "Mark it as not started" (when just completed)

---

## Category 6: Complex Real-World Scenarios

### 6.1 Software Development Projects

#### 6.1.1 Full Stack Web Application
- **Scenario**: Complete web app project planning
- **Expected breakdown**:
  - Frontend (UI components, state management, routing)
  - Backend (API endpoints, authentication, database)
  - DevOps (CI/CD, deployment, monitoring)
  - Testing (unit, integration, e2e)

#### 6.1.2 Mobile App Development
- **Scenario**: iOS/Android app project
- **Expected breakdown**:
  - Design (UI/UX, wireframes, assets)
  - Development (core features, navigation, APIs)
  - Platform-specific (iOS, Android)
  - QA and submission

#### 6.1.3 API Integration Project
- **Scenario**: Integrating with third-party APIs
- **Expected breakdown**:
  - Research (API docs, rate limits, authentication)
  - Implementation (endpoints, error handling)
  - Testing (sandbox, production)

### 6.2 Non-Software Projects

#### 6.2.1 Marketing Campaign
- **Scenario**: Plan a marketing campaign
- **Expected breakdown**:
  - Strategy (goals, audience, messaging)
  - Content (copy, visuals, landing pages)
  - Execution (scheduling, channels)
  - Analytics (tracking, reporting)

#### 6.2.2 Product Launch
- **Scenario**: New product launch planning
- **Expected breakdown**:
  - Pre-launch (beta, documentation, marketing prep)
  - Launch day (announcements, monitoring)
  - Post-launch (support, feedback, iteration)

### 6.3 Personal/Life Projects

#### 6.3.1 Learning a New Skill
- **Scenario**: User wants to learn programming
- **Expected breakdown**:
  - Fundamentals (syntax, concepts)
  - Practice (exercises, small projects)
  - Advanced (frameworks, best practices)
  - Portfolio (showcase projects)

#### 6.3.2 Event Planning
- **Scenario**: Planning a party or event
- **Expected breakdown**:
  - Planning (date, venue, guest list)
  - Logistics (catering, decorations, entertainment)
  - Communication (invitations, reminders)
  - Day-of (setup, execution, cleanup)

---

## Category 7: Tool Sequence Patterns

### 7.1 Optimal Tool Usage Patterns

**Purpose**: Teach the agent efficient tool usage sequences.

#### 7.1.1 Batch Creation
- **Pattern**: Create multiple related nodes efficiently
- **Optimal sequence**: Create parent → Create all children with parent_id
- NOT: Create child → Edit to add parent → Repeat

#### 7.1.2 Efficient Updates
- **Pattern**: Update multiple properties at once
- **Optimal**: Single `edit_task_node` with all properties
- NOT: Multiple separate `edit_task_node` calls

#### 7.1.3 Status Progression
- **Pattern**: Natural status flow during work
- **Sequence**: Create → Start working → Complete
- `create_task_node` → `update_task_status(inProgress)` → `update_task_status(completed)`

### 7.2 When NOT to Use Tools

#### 7.2.1 Information Requests Only
- **Scenario**: User just asking questions
- **Expected behavior**: Respond without tool calls
- **Examples**: "What tasks do I have?", "Explain the structure"

#### 7.2.2 Insufficient Information
- **Scenario**: Request too vague to act on
- **Expected behavior**: Ask questions, don't guess
- **Examples**: "Add a task" (what task?), "Update it" (update what?)

#### 7.2.3 After Using `finish`
- **Scenario**: Agent has completed its reasoning
- **Expected behavior**: Call `finish()` and stop tool usage

---

## Implementation Guidelines

### Training Data Generation Strategy

1. **Start with core examples** (Category 1) - ensure basic tool usage is solid
2. **Add multi-step sequences** (Category 2) - teach tool chaining
3. **Include conversational examples** (Category 3) - teach when NOT to use tools
4. **Add context-aware cases** (Category 4) - teach graph state awareness
5. **Cover edge cases** (Category 5) - teach error handling
6. **Include real-world scenarios** (Category 6) - teach domain application
7. **Reinforce patterns** (Category 7) - teach optimal sequences

### Example Counts Recommendation

| Category | Minimum Examples | Ideal Coverage |
|----------|------------------|----------------|
| 1. Single Tool | 30-50 | 5-10 per tool per scenario |
| 2. Multi-Tool | 40-60 | 10-15 per sequence type |
| 3. Conversational | 30-40 | 10+ per interaction type |
| 4. Context-Aware | 30-40 | Variety of graph states |
| 5. Error Handling | 20-30 | All error types covered |
| 6. Real-World | 20-30 | 3-5 per domain |
| 7. Patterns | 15-20 | Reinforce optimal usage |
| **Total** | **185-270** | **Comprehensive coverage** |

### Data Quality Checklist

For each example, ensure:
- [ ] Conversation history is realistic
- [ ] Graph state matches the scenario
- [ ] Tool calls are correct and efficient
- [ ] Response is natural and helpful
- [ ] Edge cases are considered
- [ ] Follow-up potential is clear

### Example Template

```python
training_examples = [
    dspy.Example(
        conversation_history="""User: I want to build a blog platform with user authentication and commenting.
Assistant: I'll help you break down this blog platform project into manageable tasks.""",
        task_nodes={
            "nodes": [],
            "links": []
        },
        response="Let me create a structured task breakdown for your blog platform project."
        # Note: During training, the trace will include the tool calls:
        # 1. create_task_node("Blog Platform", "Full-stack blog with auth and comments")
        # 2. create_task_node("User Authentication", "Login, registration, sessions", parent_id="node_1")
        # 3. create_task_node("Comment System", "Add, edit, delete comments", parent_id="node_1")
        # 4. create_task_node("Blog Posts", "CRUD for blog content", parent_id="node_1")
        # 5. finish()
    ).with_inputs("conversation_history", "task_nodes"),

    # ... more examples
]
```

---

## Appendix: Node and Tool Reference

### Node Schema
```typescript
interface Node {
  id: string;              // Unique identifier (e.g., "node_1234567890")
  name: string;            // Short task name
  description: string;     // Detailed description
  status?: 'notStarted' | 'inProgress' | 'completed';
}

interface Link {
  source: string;          // Parent node ID
  target: string;          // Child node ID
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}
```

### Tool Signatures
```python
# Create a new task node
create_task_node(
    task_name: str,           # Required: Node name
    task_description: str,    # Required: Node description
    parent_id: str = None     # Optional: Parent node ID
) -> dict

# Edit an existing task node
edit_task_node(
    node_id: str,             # Required: Node to edit
    name: str = None,         # Optional: New name
    description: str = None,  # Optional: New description
    parent_id: str = None     # Optional: New parent ("null" to remove)
) -> dict

# Update task status
update_task_status(
    node_id: str,             # Required: Node to update
    status: str               # Required: 'notStarted'|'inProgress'|'completed'
) -> dict

# Delete a task node
delete_task_node(
    node_id: str              # Required: Node to delete
) -> dict

# Signal completion
finish() -> str
```

---

## Sources and References

- [DSPy Example API](https://dspy.ai/api/primitives/Example/)
- [DSPy Data Handling](https://dspy.ai/learn/evaluation/data/)
- [DSPy Finetuning Agents Tutorial](https://dspy.ai/tutorials/games/)
- [DSPy BootstrapFinetune](https://dspy.ai/api/optimizers/BootstrapFinetune/)
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- [DSPy ReAct Module](https://dspy.ai/api/modules/ReAct/)
- [Building AI Agents with DSPy](https://dspy.ai/tutorials/customer_service_agent/)
