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
  const [showDescriptions, setShowDescriptions] = useState(true);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I can help you break down your tasks. What would you like to work on?',
      timestamp: new Date(),
    },
  ]);

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
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      // Convert messages to API format
      const apiMessages = newMessages.map(msg => ({
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

      <div className={`absolute right-0 top-0 h-full transition-transform duration-300 ease-in-out ${
        isCollapsed ? 'translate-x-full' : 'translate-x-0'
      }`}>
        <div className="w-[400px] h-full bg-background/80 backdrop-blur-sm p-6">
          <ChatInterface 
            messages={messages}
            onTaskSubmit={handleTaskSubmit}
          />
        </div>
      </div>

      <Button
        variant="ghost"
        className="absolute right-4 top-2 z-50"
        onClick={() => setIsCollapsed(!isCollapsed)}
        size="sm"
      >
        {isCollapsed ? <ChevronLeft /> : <ChevronRight />}
      </Button>
    </div>
  );
};

export default Index;