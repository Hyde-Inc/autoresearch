import React, { useCallback, useEffect, useState } from "react";
import type { RunSummary } from "../lib/api";
import { api } from "../lib/api";

function relativeTime(iso?: string | null, fallbackName?: string): string {
  let d: Date | null = null;
  if (iso) d = new Date(iso);
  else if (fallbackName && /^\d{8}-\d{6}$/.test(fallbackName)) {
    // runs dir format YYYYMMDD-HHMMSS
    const s = fallbackName;
    d = new Date(
      `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}T${s.slice(9, 11)}:${s.slice(11, 13)}:${s.slice(13, 15)}`,
    );
  }
  if (!d || Number.isNaN(d.getTime())) return fallbackName ?? "";
  const secs = Math.floor((Date.now() - d.getTime()) / 1000);
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

const STATUS_PILL: Record<string, string> = {
  running: "ae-pill--amber",
  completed: "ae-pill--green",
  stopped: "ae-pill--gray",
  unknown: "ae-pill--gray",
};

interface Props {
  taskName: string | null;
  selectedRun: string | null;
  onSelect: (run: string) => void;
  onNew?: () => void;
}

export function SessionsSidebar({ taskName, selectedRun, onSelect, onNew }: Props): React.ReactElement {
  const [runs, setRuns] = useState<RunSummary[]>([]);

  const refresh = useCallback(async () => {
    if (!taskName) {
      setRuns([]);
      return;
    }
    try {
      const index = await api.runs();
      const task = index.tasks.find((t) => t.task === taskName);
      setRuns(task ? task.runs : []);
    } catch {
      // transient
    }
  }, [taskName]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, [refresh]);

  return (
    <div className="ae-card" style={{ display: "flex", flexDirection: "column", minHeight: 0, maxHeight: "76vh" }}>
      <div style={{ padding: "12px 14px 8px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className="ae-stat__label">Sessions</span>
        {onNew && (
          <button className="ae-btn ae-btn--primary" style={{ padding: "3px 9px", fontSize: 11.5 }} onClick={onNew}>
            + New
          </button>
        )}
      </div>
      <div style={{ overflow: "auto", padding: "0 8px 10px", display: "flex", flexDirection: "column", gap: 6 }}>
        {runs.length === 0 && (
          <div style={{ fontSize: 12, color: "var(--ae-text-muted)", padding: "8px 6px" }}>
            No runs yet for this project.
          </div>
        )}
        {runs.map((r) => {
          const activeSel = r.run === selectedRun;
          const cfg = r.run_config;
          const imp = r.improvement;
          return (
            <button
              key={r.run}
              onClick={() => onSelect(r.run)}
              className="ae-card"
              style={{
                textAlign: "left",
                padding: "9px 11px",
                cursor: "pointer",
                border: activeSel ? "1px solid var(--fk-blue)" : "1px solid var(--ae-divider)",
                background: activeSel ? "var(--ae-surface-hi)" : "var(--ae-surface-elev)",
                display: "flex",
                flexDirection: "column",
                gap: 5,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: "var(--ae-text-strong)" }}>
                  {relativeTime(r.created_at, r.run)}
                </span>
                <span className={`ae-pill ${STATUS_PILL[r.status] ?? "ae-pill--gray"}`} style={{ fontSize: 10 }}>
                  {r.status === "running" && (
                    <span className="ae-pulse" style={{ width: 6, height: 6, borderRadius: 999, background: "var(--ae-warning)" }} />
                  )}
                  {r.status}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--ae-text-muted)" }}>
                {cfg && <span>{cfg.agents}a · {cfg.rounds}r</span>}
                <span>{r.attempts} exp</span>
                {r.promoted ? <span style={{ color: "var(--ae-success)" }}>★{r.promoted}</span> : null}
                {r.total_cost_usd != null && <span>${r.total_cost_usd.toFixed(2)}</span>}
              </div>
              {imp != null && (
                <div style={{ fontSize: 11.5, fontWeight: 600, color: imp > 0 ? "var(--ae-success)" : "var(--ae-text-dim)" }}>
                  {imp > 0 ? "▼" : "▲"} {Math.abs(imp * 100).toFixed(1)}% {r.primary_metric}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
