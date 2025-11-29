import React, { useCallback, useRef, useEffect, useState} from "react";
import ForceGraph2D, { ForceGraphMethods, NodeObject } from "react-force-graph-2d";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Pencil, Check, X } from "lucide-react";
import NodeActionsMenu from "./NodeActionsMenu";
//import * as d3 from "d3";

interface Node {
  id: string;
  name: string;
  val: number;
  color?: string;
  x?: number;
  y?: number;
  z?: number;
  fx?: number; 
  fy?: number;
  selected?: boolean;
  description: string;
  status?: 'notStarted' | 'inProgress' | 'completed';
}

interface Link {
  source: string;
  target: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface TaskGraphProps {
  data: GraphData;
  showDescriptions?: boolean;
  isChatOpen?: boolean;
  onNodeUpdate?: (nodeId: string, updates: Partial<Node>) => void;
}

const STATUS_COLORS = {
  notStarted: '#FFFFFF',  // White
  inProgress: '#FCD34D',  // Yellow
  completed: '#4ADE80',   // Green
} as const;

const TaskGraph = ({ data, showDescriptions = true, isChatOpen = false, onNodeUpdate }: TaskGraphProps) => {
  const fgRef = useRef<ForceGraphMethods>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedDescription, setEditedDescription] = useState("");
  const [actionMenuNode, setActionMenuNode] = useState<{x: number, y: number, node: Node} | null>(null);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState("");
  const [editingNode, setEditingNode] = useState<Node | null>(null);
  const [isAddingTask, setIsAddingTask] = useState(false);
  const [newTaskName, setNewTaskName] = useState("");
  const [newTaskDescription, setNewTaskDescription] = useState("");
  const [parentNode, setParentNode] = useState<Node | null>(null);
  const [isDeletingTask, setIsDeletingTask] = useState(false);
  const [nodeToDelete, setNodeToDelete] = useState<Node | null>(null);
  const [affectedNodes, setAffectedNodes] = useState<Node[]>([]);
  

  useEffect(() => {
    const updateDimensions = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener("resize", updateDimensions);
    updateDimensions();

    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || data.nodes.length === 0) return;
    
    // Set balanced forces for initial layout
    fg.d3Force('charge')?.strength(-800);  // Moderate repulsion
    fg.d3Force('link')
      ?.distance(200)      // Moderate distance
      ?.strength(0.3);     // Moderate link strength
    
    // Stronger center force to keep nodes in view
    fg.d3Force('center')
      ?.strength(0.5);     // Stronger centering
    
    // Set initial zoom to ensure all nodes are visible
    const nodeCount = data.nodes.length;
    const zoomLevel = Math.max(0.8, Math.min(1.2, 2.5 / Math.sqrt(nodeCount)));
    fg.zoom(zoomLevel, 1000);
    
    // Fix nodes in their positions after simulation has settled
    setTimeout(() => {
      data.nodes.forEach(node => {
        if (node.x !== undefined && node.y !== undefined) {
          node.fx = node.x;
          node.fy = node.y;
        }
      });
      fg.d3ReheatSimulation();
      
      // Find the main task node (usually the one with the most connections or specific ID)
      // Option 1: Find node with most connections (likely the main task)
      let mainNode = data.nodes[0]; // Default to first node
      let maxConnections = 0;
      
      data.nodes.forEach(node => {
        const connections = data.links.filter(link => {
          const source = typeof link.source === 'string' ? link.source : (link.source as any).id;
          const target = typeof link.target === 'string' ? link.target : (link.target as any).id;
          return source === node.id || target === node.id;
        }).length;
        
        if (connections > maxConnections) {
          maxConnections = connections;
          mainNode = node;
        }
      });
      
      // Option 2: If you know the main node has a specific ID (e.g., "main")
      const mainNodeById = data.nodes.find(node => node.id === "main");
      if (mainNodeById) {
        mainNode = mainNodeById;
      }
      
      // Calculate center offset based on chat state
      const CHAT_WIDTH = 400;
      const offsetX = isChatOpen ? +(CHAT_WIDTH / 5) : 0;
      
      // Center on the main node with offset
      if (mainNode && mainNode.x !== undefined && mainNode.y !== undefined) {
        fg.centerAt(mainNode.x + offsetX, mainNode.y, 1000);
      }
    }, 2500);
    
  }, [data, isChatOpen]);

  const handleNodeClick = useCallback((node: NodeObject) => {
    const fg = fgRef.current;
    if (!fg) return;

    setActionMenuNode(null);
    setSelectedNodeId(node.id as string);
    setSelectedNode(node as Node);

    const transitionDuration = 800;
    const CHAT_WIDTH = 400;
    const offsetX = isChatOpen ? +(CHAT_WIDTH / 5) : 0;

    // Calculate the target center position
    const x = node.x || 0;
    const y = node.y || 0;

    // Center and zoom
    const moveCamera = () => {
      fg.centerAt(x + offsetX, y, transitionDuration);
      fg.zoom(1.2, transitionDuration);  // Less zoom in

      // Adjust forces when focusing on a node
      fg.d3Force('charge')?.strength(-1500);  // Strong repulsion
      fg.d3Force('link')
        ?.distance(250)    // Large distance
        ?.strength(0.4);   // Moderate link strength

      fg.d3ReheatSimulation();
    };

    setTimeout(moveCamera, 0);
  }, [isChatOpen]);

  // Add onNodeRightClick handler
  const handleNodeRightClick = useCallback((node: NodeObject, event: MouseEvent) => {
    event.preventDefault(); // Prevent default context menu
    const fg = fgRef.current;
    if (!fg) return;
    
    // Get screen coordinates after the node is centered
    const screenPos = fg.graph2ScreenCoords(node.x || 0, node.y || 0);
    if (screenPos) {
      setActionMenuNode({
        x: screenPos.x,
        y: screenPos.y,
        node: node as Node
      });
    }
  }, []);

  const handleBackgroundClick = useCallback(() => {
    const fg = fgRef.current;
    if (!fg || isEditing) return;
    
    // Clear selections
    setActionMenuNode(null);
    setSelectedNodeId(null);
    setSelectedNode(null);

    // Calculate center offset based on chat state
    const CHAT_WIDTH = 400;
    const offsetX = isChatOpen ? +(CHAT_WIDTH / 5) : 0;

    // Reset zoom and center with offset
    fg.centerAt(offsetX, 0, 1000);
    fg.zoom(0.7, 1000);  // More zoomed out to see the whole graph

    // IMPORTANT: Don't change node positions at all
    // This preserves the current layout
    
    // Just reheat the simulation briefly
    fg.d3ReheatSimulation();
  }, [isChatOpen, isEditing]);

  const handleStartEditing = () => {
    if (selectedNode) {
      setEditedDescription(selectedNode.description);
      setIsEditing(true);
    }
  };


  const handleSaveDescription = () => {
    if (selectedNode && onNodeUpdate) {
      // Update the description through onNodeUpdate
      onNodeUpdate(selectedNode.id, { 
        description: editedDescription,
        fx: selectedNode.x,
        fy: selectedNode.y
      });
      
      // Update the local state
      setSelectedNode({ ...selectedNode, description: editedDescription });
      setIsEditing(false);
      
      // Store the current node positions to maintain them during refresh
      const nodePositions = new Map();
      data.nodes.forEach(n => {
        if (n.x !== undefined && n.y !== undefined) {
          nodePositions.set(n.id, { x: n.x, y: n.y, fx: n.x, fy: n.y });
        }
      });
      
      // Use setTimeout to ensure this runs after the state update
      setTimeout(() => {
        if (!fgRef.current) return;
        
        // Create a fresh copy of the data with proper link references
        const refreshedData = {
          nodes: [...data.nodes],
          links: [...data.links]
        };
        
        // Ensure all links use string IDs
        refreshedData.links.forEach(link => {
          if (typeof link.source === 'object' && link.source !== null) {
            link.source = (link.source as any).id || link.source;
          }
          if (typeof link.target === 'object' && link.target !== null) {
            link.target = (link.target as any).id || link.target;
          }
        });
        
        // Restore node positions
        refreshedData.nodes.forEach(n => {
          const pos = nodePositions.get(n.id);
          if (pos) {
            n.x = pos.x;
            n.y = pos.y;
            n.fx = pos.fx;
            n.fy = pos.fy;
          }
        });
        
        // Update the graph with the refreshed data
        (fgRef.current as any).graphData(refreshedData);
        fgRef.current.d3ReheatSimulation();
      }, 0);
    }
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    if (selectedNode) {
      setEditedDescription(selectedNode.description);
    }
  };

  // Add this helper function to your component
  const refreshGraphWithLinks = useCallback(() => {
    if (!fgRef.current) return;
    
    // Create a fresh copy of the data
    const refreshedData = {
      nodes: data.nodes.map(node => ({...node})),
      links: data.links.map(link => ({...link}))
    };
    
    // Update the graph with the refreshed data
    (fgRef.current as any).graphData(refreshedData);
    fgRef.current.d3ReheatSimulation();
  }, [data]);

  // Modify the cycleNodeStatus function to ensure links are properly maintained
  const cycleNodeStatus = useCallback((nodeId: string) => {
    if (!onNodeUpdate) return;
    const fg = fgRef.current;
    if (!fg) return;

    const node = data.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const statusCycle = {
      notStarted: 'inProgress',
      inProgress: 'completed',
      completed: 'notStarted'
    } as const;

    const currentStatus = node.status || 'notStarted';
    const nextStatus = statusCycle[currentStatus];

    // Update the status
    onNodeUpdate(nodeId, { 
      status: nextStatus,
      fx: node.x,
      fy: node.y
    });
    
    // Store the current node positions to maintain them during refresh
    const nodePositions = new Map();
    data.nodes.forEach(n => {
      if (n.x !== undefined && n.y !== undefined) {
        nodePositions.set(n.id, { x: n.x, y: n.y, fx: n.x, fy: n.y });
      }
    });
    
    // Use setTimeout to ensure this runs after the state update
    setTimeout(() => {
      if (!fgRef.current) return;
      
      // Create a fresh copy of the data with proper link references
      const refreshedData = {
        nodes: [...data.nodes],
        links: [...data.links]
      };
      
      // Ensure all links use string IDs
      refreshedData.links.forEach(link => {
        if (typeof link.source === 'object' && link.source !== null) {
          link.source = (link.source as any).id || link.source;
        }
        if (typeof link.target === 'object' && link.target !== null) {
          link.target = (link.target as any).id || link.target;
        }
      });
      
      // Restore node positions
      refreshedData.nodes.forEach(n => {
        const pos = nodePositions.get(n.id);
        if (pos) {
          n.x = pos.x;
          n.y = pos.y;
          n.fx = pos.fx;
          n.fy = pos.fy;
        }
      });
      
      // Update the graph with the refreshed data
      (fgRef.current as any).graphData(refreshedData);
      fgRef.current.d3ReheatSimulation();
    }, 0);
    
  }, [data.nodes, data.links, onNodeUpdate]);

  // Add these functions to handle name editing
  const handleStartEditingName = (node: Node) => {
    setEditingNode(node);
    setEditedName(node.name);
    setIsEditingName(true);
    setActionMenuNode(null); // Close the action menu
  };

  const handleSaveName = () => {
    if (editingNode && onNodeUpdate) {
      onNodeUpdate(editingNode.id, { 
        name: editedName,
        fx: editingNode.x,
        fy: editingNode.y
      });
      
      // Update the node in the data array
      const nodeIndex = data.nodes.findIndex(n => n.id === editingNode.id);
      if (nodeIndex >= 0) {
        data.nodes[nodeIndex].name = editedName;
      }
      
      setIsEditingName(false);
      setEditingNode(null);
      
      // Store the current node positions to maintain them during refresh
      const nodePositions = new Map();
      data.nodes.forEach(n => {
        if (n.x !== undefined && n.y !== undefined) {
          nodePositions.set(n.id, { x: n.x, y: n.y, fx: n.x, fy: n.y });
        }
      });
      
      // Use setTimeout to ensure this runs after the state update
      setTimeout(() => {
        if (!fgRef.current) return;
        
        // Create a fresh copy of the data with proper link references
        const refreshedData = {
          nodes: [...data.nodes],
          links: [...data.links]
        };
        
        // Ensure all links use string IDs
        refreshedData.links.forEach(link => {
          if (typeof link.source === 'object' && link.source !== null) {
            link.source = (link.source as any).id || link.source;
          }
          if (typeof link.target === 'object' && link.target !== null) {
            link.target = (link.target as any).id || link.target;
          }
        });
        
        // Restore node positions
        refreshedData.nodes.forEach(n => {
          const pos = nodePositions.get(n.id);
          if (pos) {
            n.x = pos.x;
            n.y = pos.y;
            n.fx = pos.fx;
            n.fy = pos.fy;
          }
        });
        
        // Update the graph with the refreshed data
        (fgRef.current as any).graphData(refreshedData);
        fgRef.current.d3ReheatSimulation();
      }, 0);
    }
  };

  const handleCancelEditName = () => {
    setIsEditingName(false);
    setEditingNode(null);
  };

  // Add this function to handle starting the add task process
  const handleStartAddingTask = (parentNode: Node) => {
    setParentNode(parentNode);
    setNewTaskName("");
    setNewTaskDescription("");
    setIsAddingTask(true);
    setActionMenuNode(null); // Close the action menu
  };

  // Add this function to handle saving the new task
  const handleSaveNewTask = () => {
    if (!parentNode || !onNodeUpdate) return;
    
    // Generate a unique ID for the new task
    const newTaskId = `task-${Date.now()}`;
    
    // Create the new task node
    const newTask: Node = {
      id: newTaskId,
      name: newTaskName,
      description: newTaskDescription,
      val: 1,
      status: 'notStarted',
      // Position it near the parent node with a slight offset
      x: (parentNode.x || 0) + 50,
      y: (parentNode.y || 0) + 50,
      fx: (parentNode.x || 0) + 50,
      fy: (parentNode.y || 0) + 50
    };
    
    // Create a new link from parent to the new task
    const newLink: Link = {
      source: parentNode.id,
      target: newTaskId
    };
    
    // Update the data with the new node and link
    const updatedNodes = [...data.nodes, newTask];
    const updatedLinks = [...data.links, newLink];
    
    // Call onNodeUpdate with a special action to add a new task
    // This requires modifying the parent component to handle this special case
    onNodeUpdate(parentNode.id, {
      __action: 'addSubtask',
      newTask,
      newLink
    } as any);
    
    // Close the modal
    setIsAddingTask(false);
    setParentNode(null);
    
    // Refresh the graph to ensure links are properly maintained
    setTimeout(() => {
      if (!fgRef.current) return;
      
      // Create a fresh copy of the data with proper link references
      const refreshedData = {
        nodes: updatedNodes,
        links: updatedLinks
      };
      
      // Ensure all links use string IDs
      refreshedData.links.forEach(link => {
        if (typeof link.source === 'object' && link.source !== null) {
          link.source = (link.source as any).id || link.source;
        }
        if (typeof link.target === 'object' && link.target !== null) {
          link.target = (link.target as any).id || link.target;
        }
      });
      
      // Update the graph with the refreshed data
      (fgRef.current as any).graphData(refreshedData);
      fgRef.current.d3ReheatSimulation();
    }, 0);
  };

  // Add this function to handle canceling the add task process
  const handleCancelAddTask = () => {
    setIsAddingTask(false);
    setParentNode(null);
  };

  // Fix the findDescendantNodes function
  const findDescendantNodes = useCallback((nodeId: string): Node[] => {
    // Find direct children
    const directChildren = data.links
      .filter(link => {
        if (typeof link.source === 'object') {
          return (link.source as any).id === nodeId;
        }
        return link.source === nodeId;
      })
      .map(link => {
        if (typeof link.target === 'object') {
          return (link.target as any).id;
        }
        return link.target;
      });
    
    // Find nodes corresponding to these children
    const childNodes = data.nodes.filter(node => 
      directChildren.includes(node.id)
    );
    
    // Recursively find descendants of each child
    const allDescendants = [...childNodes];
    childNodes.forEach(child => {
      const descendants = findDescendantNodes(child.id);
      allDescendants.push(...descendants);
    });
    
    return allDescendants;
  }, [data.links, data.nodes]);

  // Add this function to handle starting the delete process
  const handleStartDeletingTask = (node: Node) => {
    // Find all descendant nodes that will also be deleted
    const descendants = findDescendantNodes(node.id);
    
    setNodeToDelete(node);
    setAffectedNodes(descendants);
    setIsDeletingTask(true);
    setActionMenuNode(null); // Close the action menu
  };

  // Add this function to handle confirming deletion
  const handleConfirmDelete = () => {
    if (!nodeToDelete || !onNodeUpdate) return;
    
    // Get all node IDs to delete (the node itself and all descendants)
    const nodeIdsToDelete = [nodeToDelete.id, ...affectedNodes.map(node => node.id)];
    
    // Call onNodeUpdate with a special action to delete nodes
    onNodeUpdate(nodeToDelete.id, {
      __action: 'deleteNodes',
      nodeIds: nodeIdsToDelete
    } as any);
    
    // Close the modal
    setIsDeletingTask(false);
    setNodeToDelete(null);
    setAffectedNodes([]);
    
    // If the deleted node was selected, clear the selection
    if (selectedNodeId && nodeIdsToDelete.includes(selectedNodeId)) {
      setSelectedNodeId(null);
      setSelectedNode(null);
    }
  };

  // Add this function to handle canceling deletion
  const handleCancelDelete = () => {
    setIsDeletingTask(false);
    setNodeToDelete(null);
    setAffectedNodes([]);
  };

  return (
    <div ref={containerRef} className="w-full h-full relative bg-gradient-to-br from-gray-50 to-gray-100">
      <ForceGraph2D
        ref={fgRef}
        graphData={data}
        width={dimensions.width}
        height={dimensions.height}
        onBackgroundClick={handleBackgroundClick}
        nodeLabel="name"
        nodeColor={(node) => {
          const typedNode = node as Node;
          return STATUS_COLORS[typedNode.status || 'notStarted'];
        }}
        linkColor={() => "#e2e8f0"}
        nodeRelSize={8} 
        d3VelocityDecay={0.2}
        d3AlphaDecay={0.01}
        cooldownTime={5000}
        onEngineStop={() => {
          // Fix nodes in place after simulation stops
          data.nodes.forEach(node => {
            node.fx = node.x;
            node.fy = node.y;
          });
        }}
        onNodeClick={handleNodeClick}
        onNodeRightClick={handleNodeRightClick}
        onNodeDrag={(node) => {
          // Keep the node fixed during drag
          const typedNode = node as Node;
          typedNode.fx = typedNode.x;
          typedNode.fy = typedNode.y;
          
          // Reheat the simulation for smooth interaction
          fgRef.current?.d3ReheatSimulation();
        }}
        onNodeDragEnd={(node) => {
          // After drag ends, update the node in the data structure
          const typedNode = node as Node;
          
          // Find the node in the data array and update it
          const nodeIndex = data.nodes.findIndex(n => n.id === typedNode.id);
          if (nodeIndex >= 0) {
            // Update the node with its new position
            data.nodes[nodeIndex].x = typedNode.x;
            data.nodes[nodeIndex].y = typedNode.y;
            data.nodes[nodeIndex].fx = typedNode.x;
            data.nodes[nodeIndex].fy = typedNode.y;
          }
          
          // Use the helper function to refresh the graph
          setTimeout(refreshGraphWithLinks, 0);
        }}
        //linkDistance={250}           // Larger default link distance
        //linkStrength={0.4}          // Weaker default link strength
        enableNodeDrag={true}
        cooldownTicks={0}
        warmupTicks={100}           // Added warmup ticks
        nodeCanvasObject={(node, ctx, globalScale) => {
          const typedNode = node as Node;
          const label = typedNode.name;
          const fontSize = 14/globalScale;
          ctx.font = `${fontSize}px Inter, Sans-Serif`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.8);

          // Draw node background with status color
          ctx.fillStyle = STATUS_COLORS[typedNode.status || 'notStarted'];
          ctx.fillRect(
            node.x! - bckgDimensions[0] / 2,
            node.y! - bckgDimensions[1] / 2,
            bckgDimensions[0],
            bckgDimensions[1]
          );

          // Draw highlight if node is selected
          if (typedNode.id === selectedNodeId) {
            ctx.strokeStyle = "#4f46e5";
            ctx.lineWidth = 3/globalScale;
            ctx.strokeRect(
              node.x! - bckgDimensions[0] / 2,
              node.y! - bckgDimensions[1] / 2,
              bckgDimensions[0],
              bckgDimensions[1]
            );
          }

          // Draw the text
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = "#1a1a1a";
          ctx.fillText(label, node.x!, node.y!);
        }}
        nodeCanvasObjectMode={() => "replace"}
        nodePointerAreaPaint={(node, color, ctx, globalScale) => {
          // Define the clickable area to match the visual node rectangle
          const typedNode = node as Node;
          const label = typedNode.name;
          const fontSize = 14/globalScale;
          ctx.font = `${fontSize}px Inter, Sans-Serif`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.8);

          // Draw a filled rectangle for the clickable area
          ctx.fillStyle = color;
          ctx.fillRect(
            node.x! - bckgDimensions[0] / 2,
            node.y! - bckgDimensions[1] / 2,
            bckgDimensions[0],
            bckgDimensions[1]
          );
        }}
      />
      
      {/* Node Actions Menu */}
      {actionMenuNode && (
        <NodeActionsMenu
          x={actionMenuNode.x}
          y={actionMenuNode.y}
          onAddTask={() => handleStartAddingTask(actionMenuNode.node)}
          onDeleteTask={() => handleStartDeletingTask(actionMenuNode.node)}
          onEditTask={() => handleStartEditingName(actionMenuNode.node)}
          onChangeStatus={() => {
            cycleNodeStatus(actionMenuNode.node.id);
          }}
        />
      )}
      
      {/* Description Panel */}
      {selectedNode && showDescriptions && (
        <div className={`absolute bottom-0 left-0 p-6 bg-white/90 backdrop-blur border-t border-gray-200 transition-all duration-300 ${
          isChatOpen ? 'w-[calc(100%-400px)]' : 'w-full'
        }`}>
          <div className="max-w-4xl mx-auto">
            {isEditing ? (
              <div className="space-y-4">
                <Textarea
                  value={editedDescription}
                  onChange={(e) => setEditedDescription(e.target.value)}
                  className="min-h-[100px] text-base"
                  placeholder="Enter description..."
                  autoFocus
                />
                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCancelEdit}
                  >
                    <X className="h-4 w-4 mr-1" />
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleSaveDescription}
                  >
                    <Check className="h-4 w-4 mr-1" />
                    Save
                  </Button>
                </div>
              </div>
            ) : (
              <div className="group/description relative">
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-2 top-2 opacity-0 group-hover/description:opacity-100 transition-opacity"
                  onClick={handleStartEditing}
                >
                  <Pencil className="h-4 w-4" />
                </Button>
                <div 
                  className="bg-gray-100 rounded-lg p-4 hover:bg-gray-50 transition-colors cursor-pointer"
                  onDoubleClick={handleStartEditing}
                >
                  {selectedNode.description || "No description available"}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Name Editing Modal */}
      {isEditingName && editingNode && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 shadow-xl">
            <h3 className="text-lg font-medium mb-4">Edit Task Name</h3>
            <input
              type="text"
              value={editedName}
              onChange={(e) => setEditedName(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded mb-4"
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={handleCancelEditName}>
                Cancel
              </Button>
              <Button onClick={handleSaveName}>
                Save
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Add Task Modal */}
      {isAddingTask && parentNode && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 shadow-xl">
            <h3 className="text-lg font-medium mb-4">Add Subtask to "{parentNode.name}"</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Task Name
                </label>
                <input
                  type="text"
                  value={newTaskName}
                  onChange={(e) => setNewTaskName(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded"
                  placeholder="Enter task name"
                  autoFocus
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <Textarea
                  value={newTaskDescription}
                  onChange={(e) => setNewTaskDescription(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded min-h-[100px]"
                  placeholder="Enter task description (optional)"
                />
              </div>
            </div>
            
            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={handleCancelAddTask}>
                Cancel
              </Button>
              <Button 
                onClick={handleSaveNewTask}
                disabled={!newTaskName.trim()}
              >
                Add Task
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Task Confirmation Modal */}
      {isDeletingTask && nodeToDelete && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-[500px] shadow-xl">
            <h3 className="text-lg font-medium mb-2">Delete Task</h3>
            
            {affectedNodes.length > 0 ? (
              <div className="mb-4">
                <p className="text-red-600 font-medium">Warning: This will also delete the following subtasks:</p>
                <ul className="mt-2 max-h-[200px] overflow-y-auto border border-gray-200 rounded p-2 bg-gray-50">
                  {affectedNodes.map(node => (
                    <li key={node.id} className="py-1 border-b border-gray-100 last:border-b-0">
                      {node.name}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="mb-4">Are you sure you want to delete "{nodeToDelete.name}"?</p>
            )}
            
            <div className="flex justify-end gap-2 mt-6">
              <Button variant="outline" onClick={handleCancelDelete}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleConfirmDelete}>
                Delete{affectedNodes.length > 0 ? ' All' : ''}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TaskGraph;