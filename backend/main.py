from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union
import json
import asyncio
import logging
from llm_model.agent import Agent

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI(title="Task Graph API")

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI agent
agent = Agent()

# Request/Response models
class Message(BaseModel):
    id: str
    type: str  # 'user' or 'assistant'
    content: str
    timestamp: str

class Node(BaseModel):
    model_config = ConfigDict(extra='allow')

    id: str
    name: str
    description: str
    status: Optional[str] = None

class Link(BaseModel):
    model_config = ConfigDict(extra='allow')

    source: Union[str, Dict[str, Any]]
    target: Union[str, Dict[str, Any]]

    @field_validator('source', 'target', mode='before')
    @classmethod
    def extract_id_from_node(cls, v):
        """Extract ID from node object if it's a dict."""
        if isinstance(v, dict) and 'id' in v:
            return v['id']
        return v

class GraphData(BaseModel):
    model_config = ConfigDict(extra='allow')

    nodes: List[Node]
    links: List[Link]

class ChatRequest(BaseModel):
    chatHistory: List[Message]
    graph: GraphData

class ChatResponse(BaseModel):
    message_response: str
    graph_data: GraphData


def convert_chat_history_to_openai_messages(chat_history: List[Message]) -> List[Dict[str, str]]:
    """Convert custom chat format to OpenAI message format."""
    openai_messages = []
    for message in chat_history:
        role = message.type  # 'user' or 'assistant'
        content = message.content
        openai_messages.append({"role": role, "content": content})
    return openai_messages


def parse_graph_data(graph_data: GraphData) -> Dict[str, Any]:
    """Parse and simplify graph data structure."""
    return {
        'nodes': [node.model_dump() for node in graph_data.nodes],
        'links': [link.model_dump() for link in graph_data.links]
    }


async def generate_streaming_response(chat_history: List[Message], graph_data: GraphData):
    """
    Generate streaming response with both text chunks and final graph data.

    Yields:
        - Text chunks as they're generated
        - Final graph data when complete
    """
    # Convert inputs
    messages = convert_chat_history_to_openai_messages(chat_history)
    graph = parse_graph_data(graph_data)

    # Stream the agent's response
    async for chunk in agent.query_stream(messages, graph):
        # Yield each chunk as Server-Sent Event
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send done signal
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Task Graph API is running"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Non-streaming endpoint for backward compatibility.
    Returns the complete response at once.
    """
    messages = convert_chat_history_to_openai_messages(request.chatHistory)
    graph = parse_graph_data(request.graph)

    result = agent.query(messages, graph)

    return ChatResponse(
        message_response=str(result.response),
        graph_data=GraphData(
            nodes=[Node(**node) for node in graph['nodes']],
            links=[Link(**link) for link in graph['links']]
        )
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Streams the AI response token by token for better UX.
    """
    return StreamingResponse(
        generate_streaming_response(request.chatHistory, request.graph),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering for nginx
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
