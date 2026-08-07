import React from "react";
import type { Attempt } from "../lib/api";
import { categoryPill } from "../lib/api";

function elapsed(iso: string): string {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
}

export function LiveAgents({
  attempts,
  onSelect,
}: {
  attempts: Attempt[];
  onSelect: (id: string) => void;
}): React.ReactElement {
  const running = attempts.filter((a) => a.status === "running");
  return (
    <div className="ae-card" style={{ display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "12px 16px 8px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className="ae-stat__label">Live agents</span>
        <span className={`ae-pill ${running.length > 0 ? "ae-pill--amber" : "ae-pill--gray"}`}>
          {running.length} active
        </span>
      </div>
      <div style={{ padding: "0 10px 12px", display: "flex", flexDirection: "column", gap: 8 }}>
        {running.length === 0 && (
          <div style={{ fontSize: 12, color: "var(--ae-text-muted)", padding: "8px 6px" }}>
            No agents running. Start a run to watch experiments live.
          </div>
        )}
        {running.map((attempt) => (
          <button
            key={attempt.id}
            onClick={() => onSelect(attempt.id)}
            style={{
              textAlign: "left",
              background: "var(--ae-surface-elev)",
              border: "1px solid var(--ae-divider)",
              borderRadius: "var(--ae-radius-sm)",
              padding: "10px 12px",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              gap: 6,
              color: "var(--ae-text)",
              fontFamily: "inherit",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                className="ae-pulse"
                style={{ width: 8, height: 8, borderRadius: 999, background: "var(--ae-warning)", boxShadow: "0 0 6px var(--ae-warning)", flexShrink: 0 }}
              />
              <span style={{ fontSize: 12.5, fontWeight: 600, color: "var(--ae-text-strong)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {attempt.idea.title}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="mono" style={{ fontSize: 10.5, color: "var(--ae-text-muted)" }}>{attempt.id}</span>
              <span className={`ae-pill ${categoryPill(attempt.idea.category)}`} style={{ fontSize: 10 }}>
                {attempt.idea.category}
              </span>
              <span className="tabular" style={{ fontSize: 10.5, color: "var(--ae-text-dim)", marginLeft: "auto" }}>
                round {attempt.round} · {elapsed(attempt.created_at)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
