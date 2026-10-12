import React from "react";

export default function CampusSketchBG() {
  return (
    <div
      className="campus-sketch-overlay"
      aria-hidden="true"
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
        opacity: 0.12,
        overflow: "hidden",
      }}
    >
      <svg
        width="100%"
        height="100%"
        xmlns="http://www.w3.org/2000/svg"
        style={{ position: "absolute", top: 0, left: 0 }}
      >
        <defs>
          <pattern id="campus-grid" width="60" height="60" patternUnits="userSpaceOnUse">
            <path d="M 60 0 L 0 0 0 60" fill="none" stroke="currentColor" strokeWidth="0.8" strokeDasharray="3 3" />
            <circle cx="0" cy="0" r="1.5" fill="currentColor" opacity="0.6" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#campus-grid)" />
        {/* Subtle decorative campus architectural contours */}
        <g stroke="currentColor" fill="none" strokeWidth="1" opacity="0.4">
          <path d="M 100,200 L 180,140 L 260,200 L 260,320 L 100,320 Z" />
          <path d="M 180,140 L 180,90 L 195,90 L 195,140" />
          <path d="M 850,150 L 980,150 L 1020,230 L 980,310 L 850,310 Z" />
          <line x1="880" y1="180" x2="880" y2="280" />
          <line x1="920" y1="180" x2="920" y2="280" />
          <line x1="960" y1="180" x2="960" y2="280" />
          <circle cx="500" cy="700" r="120" strokeDasharray="6 6" />
          <circle cx="500" cy="700" r="80" />
          <line x1="360" y1="700" x2="640" y2="700" />
          <line x1="500" y1="560" x2="500" y2="840" />
        </g>
      </svg>
    </div>
  );
}

