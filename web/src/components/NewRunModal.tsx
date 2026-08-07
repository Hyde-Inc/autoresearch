import React, { useState } from "react";
import { api } from "../lib/api";

export function NewRunModal({ onClose, onStarted }: { onClose: () => void; onStarted: () => void }): React.ReactElement {
  const [goal, setGoal] = useState("Reduce WMAPE on the 28-day validation window");
  const [parallel, setParallel] = useState(4);
  const [maxExperiments, setMaxExperiments] = useState(8);
  const [busy, setBusy] = useState(false);

  const start = async (): Promise<void> => {
    setBusy(true);
    try {
      await api.start({ goal, parallel, max_experiments: maxExperiments });
      onStarted();
      onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(4,6,10,0.7)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 50,
      }}
      onClick={onClose}
    >
      <div
        className="ae-card ae-fade-in"
        style={{ width: 460, padding: 20, display: "flex", flexDirection: "column", gap: 14 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <div style={{ fontFamily: "var(--ae-font-display)", fontSize: 16, fontWeight: 600, color: "var(--ae-text-strong)" }}>
            Start a research run
          </div>
          <div style={{ fontSize: 12, color: "var(--ae-text-muted)", marginTop: 2 }}>
            The director proposes ideas; parallel agents implement and test them.
          </div>
        </div>
        <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="ae-stat__label">Goal</span>
          <textarea className="ae-textarea" rows={2} value={goal} onChange={(e) => setGoal(e.target.value)} style={{ resize: "vertical" }} />
        </label>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span className="ae-stat__label">Parallel agents</span>
            <input className="ae-input" type="number" min={1} max={8} value={parallel} onChange={(e) => setParallel(Number(e.target.value))} />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span className="ae-stat__label">Max experiments</span>
            <input className="ae-input" type="number" min={1} max={32} value={maxExperiments} onChange={(e) => setMaxExperiments(Number(e.target.value))} />
          </label>
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button className="ae-btn" onClick={onClose}>Cancel</button>
          <button className="ae-btn ae-btn--primary" onClick={start} disabled={busy}>
            {busy ? "Starting…" : "Launch agents"}
          </button>
        </div>
      </div>
    </div>
  );
}
