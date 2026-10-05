import React, { useState, useEffect, useRef } from 'react';
import { Network, Activity, Clock, ShieldAlert } from 'lucide-react';

export default function PatientGraphCanvas({ nodes = [], edges = [], patientId, readmissionRisk }) {
  const canvasRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodePositions, setNodePositions] = useState({});

  useEffect(() => {
    if (!nodes || nodes.length === 0) return;

    // Calculate dynamic layout positions (layered by relative_time_hours)
    const sortedNodes = [...nodes].sort((a, b) => a.relative_time_hours - b.relative_time_hours);
    const canvas = canvasRef.current;
    if (!canvas) return;

    const width = canvas.clientWidth || 700;
    const height = 360;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');

    // Layout math: distribute nodes horizontally by time, vertically with offsets
    const minTime = Math.min(...sortedNodes.map(n => n.relative_time_hours));
    const maxTime = Math.max(...sortedNodes.map(n => n.relative_time_hours)) || (minTime + 1);

    const positions = {};
    const paddingX = 70;
    const paddingY = 60;
    const usableWidth = width - paddingX * 2;
    const usableHeight = height - paddingY * 2;

    sortedNodes.forEach((node, index) => {
      const timeRatio = maxTime === minTime ? index / (sortedNodes.length || 1) : (node.relative_time_hours - minTime) / (maxTime - minTime);
      const x = paddingX + timeRatio * usableWidth;
      // Stagger Y positions to avoid overlap
      const yOffset = (index % 3 - 1) * (usableHeight * 0.35);
      const y = height / 2 + yOffset;
      positions[node.id] = { x, y, node };
    });

    setNodePositions(positions);

    // Draw function
    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw background grid lines & time ticks
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      for (let x = paddingX; x <= width - paddingX; x += 100) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      // Draw Directed Edges
      edges.forEach(edge => {
        const p1 = positions[edge.source];
        const p2 = positions[edge.target];
        if (!p1 || !p2) return;

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const angle = Math.atan2(dy, dx);
        const radius = 22;

        const startX = p1.x + Math.cos(angle) * radius;
        const startY = p1.y + Math.sin(angle) * radius;
        const endX = p2.x - Math.cos(angle) * radius;
        const endY = p2.y - Math.sin(angle) * radius;

        // Gradient line for temporal flow
        const gradient = ctx.createLinearGradient(startX, startY, endX, endY);
        gradient.addColorStop(0, 'rgba(56, 189, 248, 0.6)');
        gradient.addColorStop(1, 'rgba(99, 102, 241, 0.8)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        // Curve control point
        const cpX = (startX + endX) / 2;
        const cpY = (startY + endY) / 2 - 15;
        ctx.quadraticCurveTo(cpX, cpY, endX, endY);
        ctx.stroke();

        // Arrow head
        ctx.fillStyle = '#6366F1';
        ctx.beginPath();
        ctx.arc(endX, endY, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw Nodes
      Object.values(positions).forEach(({ x, y, node }) => {
        const isHovered = hoveredNode && hoveredNode.id === node.id;
        const isSelected = selectedNode && selectedNode.id === node.id;

        // Severity Color
        let nodeColor = '#38BDF8';
        if (node.severity === 'severe') nodeColor = '#EF4444';
        else if (node.severity === 'moderate') nodeColor = '#F59E0B';

        // Outer glow
        if (isHovered || isSelected) {
          ctx.beginPath();
          ctx.arc(x, y, 28, 0, Math.PI * 2);
          ctx.fillStyle = nodeColor === '#EF4444' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(56, 189, 248, 0.3)';
          ctx.fill();
        }

        // Node Circle
        ctx.beginPath();
        ctx.arc(x, y, isHovered ? 22 : 18, 0, Math.PI * 2);
        ctx.fillStyle = '#0F172A';
        ctx.fill();
        ctx.lineWidth = isHovered ? 3 : 2;
        ctx.strokeStyle = nodeColor;
        ctx.stroke();

        // Inner core
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = nodeColor;
        ctx.fill();

        // Text Code Label
        ctx.fillStyle = '#F8FAFC';
        ctx.font = '600 11px Outfit, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(node.code, x, y - 26);

        // Time Onset Subtitle
        ctx.fillStyle = '#94A3B8';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText(`+${node.relative_time_hours.toFixed(1)}h`, x, y + 32);
      });
    };

    draw();

  }, [nodes, edges, hoveredNode, selectedNode]);

  // Handle Mouse Hover & Click
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let found = null;
    Object.values(nodePositions).forEach(({ x, y, node }) => {
      const dist = Math.hypot(mouseX - x, mouseY - y);
      if (dist <= 25) {
        found = node;
      }
    });
    setHoveredNode(found);
  };

  const handleClick = () => {
    if (hoveredNode) {
      setSelectedNode(hoveredNode);
    }
  };

  const activeNode = selectedNode || hoveredNode || (nodes.length > 0 ? nodes[0] : null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ position: 'relative', width: '100%', background: 'rgba(15, 23, 42, 0.6)', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', overflow: 'hidden' }}>
        <div style={{ padding: '12px 18px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Network size={18} color="#38BDF8" />
            <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#F8FAFC' }}>
              Dynamic Temporal Event DAG Topology
            </span>
          </div>
          <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
            Hover/click nodes for clinical event attributes
          </span>
        </div>

        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onClick={handleClick}
          style={{ width: '100%', height: '360px', cursor: hoveredNode ? 'pointer' : 'default', display: 'block' }}
        />

        {/* Legend Overlay */}
        <div style={{ position: 'absolute', bottom: '12px', left: '16px', display: 'flex', gap: '14px', background: 'rgba(15, 23, 42, 0.85)', padding: '6px 12px', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#94A3B8' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#EF4444' }}></span> Severe
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#94A3B8' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#F59E0B' }}></span> Moderate
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: '#94A3B8' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#38BDF8' }}></span> Mild
          </div>
        </div>
      </div>

      {/* Selected/Hovered Node Details Banner */}
      {activeNode && (
        <div style={{ padding: '14px 18px', background: 'rgba(30, 41, 59, 0.7)', borderRadius: '12px', border: '1px solid rgba(56, 189, 248, 0.2)', display: 'flex', flexWrap: 'wrap', gap: '20px', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className={`badge ${activeNode.severity === 'severe' ? 'badge-danger' : activeNode.severity === 'moderate' ? 'badge-warning' : 'badge-info'}`}>
                {activeNode.severity}
              </span>
              <span style={{ fontWeight: 700, color: '#38BDF8', fontSize: '1rem', fontFamily: 'Outfit' }}>
                ICD-10: {activeNode.code}
              </span>
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC', marginTop: '4px' }}>
              {activeNode.label}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Clock size={16} color="#94A3B8" />
              <div>
                <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>Onset Latency</div>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#F8FAFC' }}>+{activeNode.relative_time_hours.toFixed(1)}h Post-ICU</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={16} color="#94A3B8" />
              <div>
                <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>Encounter Reference</div>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#F8FAFC' }}>{activeNode.encounter_id || 'ENC-ICU-MAIN'}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
