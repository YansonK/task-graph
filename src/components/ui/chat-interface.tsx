import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar } from "@/components/ui/avatar";
import { Bot, User } from "lucide-react";
import ThinkingIndicator from "@/components/ui/thinking-indicator";

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isThinking?: boolean;
  thinking?: string;
}

interface ChatInterfaceProps {
  messages: Message[];
  onTaskSubmit: (task: string) => void;
}

const ChatInterface = ({ messages, onTaskSubmit }: ChatInterfaceProps) => {
  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-1 pr-4">
        <div className="space-y-4 pb-4">
          {messages.map((message) => (
            <div key={message.id} className="space-y-2">
              {/* Thinking Display (only for assistant messages with thinking content) */}
              {message.type === 'assistant' && message.thinking && (
                <div className="ml-11 mr-auto max-w-[80%]">
                  <div className="p-2 bg-muted/30 rounded-md border border-muted/50">
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></div>
                      <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                        Thinking
                      </span>
                    </div>
                    <div className="text-[11px] text-muted-foreground/60 font-mono whitespace-pre-wrap max-h-24 overflow-y-auto leading-relaxed">
                      {message.thinking}
                    </div>
                  </div>
                </div>
              )}

              {/* Message Content */}
              <div
                className={`flex gap-3 ${
                  message.type === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.type === 'assistant' && (
                  <Avatar className="h-8 w-8">
                    <Bot className="h-4 w-4" />
                  </Avatar>
                )}

                <div
                  className={`rounded-lg px-4 py-2 max-w-[80%] ${
                    message.type === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  }`}
                >
                  {message.isThinking ? (
                    <ThinkingIndicator />
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  )}
                  <span className="text-xs opacity-70 mt-1 block">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>

                {message.type === 'user' && (
                  <Avatar className="h-8 w-8">
                    <User className="h-4 w-4" />
                  </Avatar>
                )}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <TaskInput onSubmit={onTaskSubmit} />
    </div>
  );
};

const TaskInput = ({ onSubmit }) => {
  const [input, setInput] = React.useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSubmit(input);
      setInput('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter your task here..."
        className="w-full px-4 py-2 rounded-lg bg-background border focus:outline-none focus:ring-2 focus:ring-primary"
      />
    </form>
  );
};

export default ChatInterface;