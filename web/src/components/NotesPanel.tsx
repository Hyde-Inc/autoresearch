import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Note } from "../lib/api";

export function NotesPanel({ notes }: { notes: Note[] }): React.ReactElement {
  const [expanded, setExpanded] = useState<number | null>(notes.length > 0 ? notes.length - 1 : null);
  return (
    <div className="ae-card" style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
      <div style={{ padding: "12px 16px 8px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className="ae-stat__label">Research director notes</span>
        <span className="ae-pill ae-pill--blue">{notes.length} reflections</span>
      </div>
      <div style={{ padding: "0 10px 12px", overflow: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
        {notes.length === 0 && (
          <div style={{ fontSize: 12, color: "var(--ae-text-muted)", padding: "8px 6px" }}>
            The director writes a reflection after each round — what worked, what failed, and why.
          </div>
        )}
        {notes.map((note, i) => (
          <div key={i} style={{ background: "var(--ae-surface-elev)", border: "1px solid var(--ae-divider)", borderRadius: "var(--ae-radius-sm)" }}>
            <button
              onClick={() => setExpanded(expanded === i ? null : i)}
              style={{
                width: "100%",
                textAlign: "left",
                background: "transparent",
                border: "none",
                color: "var(--ae-text-strong)",
                fontSize: 12.5,
                fontWeight: 600,
                padding: "9px 12px",
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                fontFamily: "inherit",
              }}
            >
              {note.heading}
              <span style={{ color: "var(--ae-text-dim)" }}>{expanded === i ? "▾" : "▸"}</span>
            </button>
            {expanded === i && (
              <div className="ae-note-body ae-fade-in" style={{ padding: "0 12px 10px", maxHeight: 380, overflow: "auto" }}>
                <ReactMarkdown>{note.body}</ReactMarkdown>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
