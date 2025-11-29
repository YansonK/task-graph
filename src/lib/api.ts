// API client for FastAPI backend

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface Node {
  id: string;
  name: string;
  description: string;
  status?: 'notStarted' | 'inProgress' | 'completed';
}

export interface Link {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: Node[];
  links: Link[];
}

export interface ChatRequest {
  chatHistory: Message[];
  graph: GraphData;
}

export interface ChatResponse {
  message_response: string;
  graph_data: GraphData;
}

/**
 * Non-streaming chat endpoint
 */
export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Streaming chat endpoint using Server-Sent Events (SSE)
 *
 * @param request - Chat request with history and graph
 * @param onToken - Callback for each token received
 * @param onThinking - Callback for each thinking token received
 * @param onGraphUpdate - Callback when graph is updated
 * @param onDone - Callback when streaming completes
 */
export async function sendMessageStream(
  request: ChatRequest,
  onToken: (token: string) => void,
  onThinking: (token: string) => void,
  onGraphUpdate: (graphData: GraphData) => void,
  onDone: () => void
): Promise<void> {
  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6); // Remove 'data: ' prefix

          try {
            const parsed = JSON.parse(data);

            switch (parsed.type) {
              case 'token':
                onToken(parsed.content);
                break;
              case 'thinking':
                onThinking(parsed.content);
                break;
              case 'graph_update':
                onGraphUpdate(parsed.graph_data);
                break;
              case 'done':
                onDone();
                return;
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Health check endpoint
 */
export async function healthCheck(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_URL}/`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}
