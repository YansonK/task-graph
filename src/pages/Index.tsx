import React, { useState } from "react";
import TaskGraph from "@/components/TaskGraph";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import ChatInterface from "@/components/ui/chat-interface";
import DescriptionToggle from "@/components/ui/descriptiontoggle";
import { sendMessageStream, GraphData as APIGraphData } from "@/lib/api";

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isThinking?: boolean;
  thinking?: string; // Agent's reasoning process
}

interface Link {
  source: string;
  target: string;
}

interface Node {
  id: string;
  name: string;
  val: number;
  color?: string;
  x?: number;
  y?: number;
  z?: number;
  selected?: boolean;
  description: string;
  status?: 'notStarted' | 'inProgress' | 'completed';
}

const Index = () => {
  const [graphData, setGraphData] = useState({
    nodes: [],
    links: [],
  });

  const [isCollapsed, setIsCollapsed] = useState(true);
  const [chatWidth, setChatWidth] = useState(400); // Current chat width
  const [savedChatWidth, setSavedChatWidth] = useState(400); // Width before collapse
  const [isResizing, setIsResizing] = useState(false);
  const [showDescriptions, setShowDescriptions] = useState(true);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I can help you break down your tasks. What would you like to work on?',
      timestamp: new Date(),
    },
  ]);

  // Handle resize drag
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      const newWidth = window.innerWidth - e.clientX;
      // Minimum width is 400px, maximum is 80% of screen
      const clampedWidth = Math.max(400, Math.min(newWidth, window.innerWidth * 0.8));
      setChatWidth(clampedWidth);
      setSavedChatWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Handle collapse/expand with width memory
  const toggleChat = () => {
    if (isCollapsed) {
      // Expanding - restore saved width
      setChatWidth(savedChatWidth);
      setIsCollapsed(false);
    } else {
      // Collapsing - save current width
      setSavedChatWidth(chatWidth);
      setIsCollapsed(true);
    }
  };

  const handleTaskSubmit = async (task: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: task,
      timestamp: new Date(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);

    // Create a placeholder assistant message that will be updated with streaming content
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isThinking: true,
      thinking: '', // Initialize thinking content
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      // Convert messages to API format (exclude thinking field and only send completed messages)
      const apiMessages = newMessages
        .filter(msg => !msg.isThinking && msg.content) // Only send non-thinking messages with content
        .map(msg => ({
          id: msg.id,
          type: msg.type,
          content: msg.content,
          timestamp: msg.timestamp.toISOString(),
        }));

      // Send streaming request
      await sendMessageStream(
        {
          chatHistory: apiMessages,
          graph: graphData,
        },
        // onToken: Append each token to the assistant message
        (token: string) => {
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, content: msg.content + token, isThinking: false }
                : msg
            )
          );
        },
        // onThinking: Append thinking content to the message
        (thinking: string) => {
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, thinking: (msg.thinking || '') + thinking }
                : msg
            )
          );
        },
        // onReplaceResponse: Replace the entire response content
        (content: string) => {
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, content: content, isThinking: false }
                : msg
            )
          );
        },
        // onGraphUpdate: Update the graph data
        (updatedGraph: APIGraphData) => {
          console.log("Graph data updated:", updatedGraph);
          setGraphData(updatedGraph);
        },
        // onDone: Streaming complete
        () => {
          console.log("Streaming complete");
        }
      );
    } catch (error) {
      console.error("Error processing task:", error);

      // Clear thinking state and show error message
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...msg, content: 'Sorry, there was an error processing your request. Please try again.', isThinking: false }
            : msg
        )
      );
    }
  };
  const handleNodeUpdate = (nodeId: string, updates: any) => {
    // Check if this is a special action
    if (updates.__action === 'addSubtask') {
      // Extract the new task and link from the updates
      const { newTask, newLink } = updates;
      
      // Update the graph data to include the new node and link
      setGraphData(prevData => ({
        nodes: [...prevData.nodes, newTask],
        links: [...prevData.links, newLink]
      }));
    } else if (updates.__action === 'deleteNodes') {
      // Extract the node IDs to delete
      const { nodeIds } = updates;
      
      // Update the graph data to remove the nodes and their links
      setGraphData(prevData => ({
        nodes: prevData.nodes.filter(node => !nodeIds.includes(node.id)),
        links: prevData.links.filter(link => 
          !nodeIds.includes(typeof link.source === 'string' ? link.source : link.source.id) && 
          !nodeIds.includes(typeof link.target === 'string' ? link.target : link.target.id)
        )
      }));
    } else {
      // Handle regular node updates (status, name, description, etc.)
      setGraphData(prevData => ({
        ...prevData,
        nodes: prevData.nodes.map(node => 
          node.id === nodeId ? { ...node, ...updates } : node
        )
      }));
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden dark relative">
      <div className="absolute inset-0">
        <TaskGraph 
          data={graphData} 
          showDescriptions={showDescriptions}
          isChatOpen={!isCollapsed}
          onNodeUpdate={handleNodeUpdate}
        />
        <DescriptionToggle 
          showDescriptions={showDescriptions}
          onToggle={setShowDescriptions}
        />
      </div>

      <div
        className={`absolute right-0 top-0 h-full transition-transform duration-300 ease-in-out ${
          isCollapsed ? 'translate-x-full' : 'translate-x-0'
        }`}
        style={{ width: `${chatWidth}px` }}
      >
        {/* Resize Handle */}
        {!isCollapsed && (
          <div
            onMouseDown={handleMouseDown}
            className={`absolute left-0 top-0 h-full w-2 cursor-ew-resize flex items-center justify-center hover:bg-muted/50 transition-colors ${
              isResizing ? 'bg-muted/50' : ''
            }`}
            style={{ zIndex: 60 }}
          >
            <div className="w-1 h-12 bg-muted-foreground/20 rounded-full" />
          </div>
        )}

        <div className="h-full bg-background/80 backdrop-blur-sm p-6 flex flex-col">
          <ChatInterface
            messages={messages}
            onTaskSubmit={handleTaskSubmit}
          />
        </div>
      </div>

      <Button
        variant="ghost"
        className="absolute right-4 top-2 z-50"
        onClick={toggleChat}
        size="sm"
      >
        {isCollapsed ? <ChevronLeft /> : <ChevronRight />}
      </Button>
    </div>
  );
};

export default Index;