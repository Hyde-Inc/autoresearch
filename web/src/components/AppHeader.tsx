import React from "react";
import type { RunState, TaskRuns } from "../lib/api";

interface Props {
  tasks: TaskRuns[];
  selectedTask: string | null;
  selectedRun: string | null;
  state: RunState;
  onSelect: (task: string, run: string) => void;
  onNewRun: () => void;
  onStop: () => void;
  onDefineMetric: () => void;
}

export function AppHeader(props: Props): React.ReactElement {
  const { tasks, selectedTask, selectedRun, state, onSelect, onNewRun, onStop, onDefineMetric } = props;
  const running = state.status === "running";
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 24,
        padding: "11px 20px",
        background: "var(--ae-surface)",
        borderBottom: "1px solid var(--ae-divider)",
        flex: "0 0 auto",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14, minWidth: 0 }}>
        <div
          style={{
            width: 34,
            height: 34,
            borderRadius: 8,
            background: "linear-gradient(135deg, var(--fk-blue) 0%, #7c3aed 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "var(--ae-font-display)",
            fontWeight: 700,
            fontSize: 16,
            color: "#fff",
          }}
        >
          A
        </div>
        <div style={{ width: 1, height: 26, background: "var(--ae-divider)" }} />
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontFamily: "var(--ae-font-display)",
              fontSize: 15,
              fontWeight: 600,
              letterSpacing: "-0.02em",
              color: "var(--ae-text-strong)",
              lineHeight: 1.2,
            }}
          >
            Autoresearch Lab
          </div>
          <div style={{ fontSize: 11.5, color: "var(--ae-text-muted)", marginTop: 1 }}>
            Parallel ML experimentation · {selectedTask ?? "no task"}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <select
          className="ae-select"
          value={selectedTask && selectedRun ? `${selectedTask}/${selectedRun}` : ""}
          onChange={(e) => {
            const [task, run] = e.target.value.split("/");
            onSelect(task, run);
          }}
        >
          {tasks.flatMap((t) =>
            t.runs.map((r) => (
              <option key={`${t.task}/${r.run}`} value={`${t.task}/${r.run}`}>
                {t.task} · {r.run} ({r.attempts} experiments)
              </option>
            )),
          )}
        </select>
        <div
          className={`ae-pill ${running ? "ae-pill--green" : "ae-pill--gray"}`}
          title="Run status"
        >
          <span
            className={running ? "ae-pulse" : undefined}
            style={{
              width: 7,
              height: 7,
              borderRadius: 999,
              background: running ? "#3ec27a" : "#8893a8",
              boxShadow: running ? "0 0 6px #3ec27a" : "none",
            }}
          />
          {running ? "run in progress" : (state.status ?? "idle")}
        </div>
        <button className="ae-btn" onClick={onDefineMetric} disabled={running} title={running ? "Finish the current run before changing the metric" : "Define the evaluation metric in plain English"}>
          Define metric
        </button>
        {running ? (
          <button className="ae-btn ae-btn--danger" onClick={onStop}>
            Stop after round
          </button>
        ) : (
          <button className="ae-btn ae-btn--primary" onClick={onNewRun}>
            New run
          </button>
        )}
      </div>
    </header>
  );
}
