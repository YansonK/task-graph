import React from 'react';

const ThinkingIndicator = () => {
  return (
    <div className="flex items-center gap-1">
      <span className="text-sm text-muted-foreground">Thinking</span>
      <div className="flex gap-1">
        <span className="animate-bounce [animation-delay:0ms] text-muted-foreground">.</span>
        <span className="animate-bounce [animation-delay:150ms] text-muted-foreground">.</span>
        <span className="animate-bounce [animation-delay:300ms] text-muted-foreground">.</span>
      </div>
    </div>
  );
};

export default ThinkingIndicator;
