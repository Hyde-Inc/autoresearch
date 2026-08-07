import React, { useCallback, useEffect, useState } from "react";
import { AttemptsTable } from "../components/AttemptsTable";
import { ExperimentChart } from "../components/ExperimentChart";
import { KpiStrip } from "../components/KpiStrip";
import { LiveAgents } from "../components/LiveAgents";
import { NotesPanel } from "../components/NotesPanel";
import type { Attempt, Note, RunState } from "../lib/api";
import { api } from "../lib/api";
import { StepShell } from "./StepShell";

const POLL_MS = 2500;

interface Props {
  config: string;
  taskName: string;
  onBack: () => void;
  onDone: () => void;
  onRun: (task: string, run: string) => void;
}

export function StepRun({ config, taskName, onBack, onDone, onRun }: Props): React.ReactElement {
  const [run, setRun] = useState<string | null>(null);
  const [state, setState] = useState<RunState>({});
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

  const [goal, setGoal] = useState("");
  const [parallel, setParallel] = useState(3);
  const [maxExperiments, setMaxExperiments] = useState(8);

  const refresh = useCallback(async () => {
    try {
      const index = await api.runs();
      const task = index.tasks.find((t) => t.task === taskName);
      if (!task || task.runs.length === 0) {
        setRun(null);
        setAttempts([]);
        return;
      }
      let chosen = run;
      if (!chosen) {
        const running = task.runs.find((r) => r.status === "running");
        chosen = (running ?? task.runs[0]).run;
        setRun(chosen);
      }
      if (chosen) {
        onRun(taskName, chosen);
        const [s, a, n] = await Promise.all([
          api.state(taskName, chosen),
          api.attempts(taskName, chosen),
          api.notes(taskName, chosen),
        ]);
        setState(s);
        setAttempts(a);
        setNotes(n);
      }
    } catch {
      // transient
    }
  }, [taskName, run, onRun]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  const launch = async (): Promise<void> => {
    setLaunching(true);
    try {
      await api.start({
        config,
        goal: goal.trim() || undefined,
        parallel,
        max_experiments: maxExperiments,
      });
      setRun(null); // let refresh pick up the newly started run
      setTimeout(refresh, 1500);
    } finally {
      setLaunching(false);
    }
  };

  const running = state.status === "running";
  const metric = state.primary_metric ?? "wmape";
  const baseline = state.baseline?.[metric];
  const hasRun = run !== null;

  return (
    <StepShell
      title="Trigger the Autoresearch run"
      description={
        <>
          The director proposes diverse experiments; parallel agents implement and test each one
          against your metric and the hidden holdout. Winners are promoted automatically. Watch it
          unfold live below. <span style={{ color: "var(--ae-text-dim)" }}>Demo limits: up to 4 parallel agents and 4 rounds.</span>
        </>
      }
      onBack={onBack}
      onNext={hasRun ? onDone : undefined}
      nextLabel="See learnings →"
      wide
    >
      <div className="ae-card ae-card--padded" style={{ display: "grid", gridTemplateColumns: "2fr 0.7fr 0.7fr auto", gap: 14, alignItems: "end" }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="ae-stat__label">Goal (optional override)</span>
          <input className="ae-input" value={goal} placeholder={state.goal || "Reduce error while keeping runtime and holdout stable"} onChange={(e) => setGoal(e.target.value)} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="ae-stat__label">Parallel agents (max 4)</span>
          <input
            className="ae-input"
            type="number"
            min={1}
            max={4}
            value={parallel}
            onChange={(e) => setParallel(Math.min(4, Math.max(1, Number(e.target.value))))}
          />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="ae-stat__label">Max experiments</span>
          <input className="ae-input" type="number" min={1} max={16} value={maxExperiments} onChange={(e) => setMaxExperiments(Math.min(16, Math.max(1, Number(e.target.value))))} />
        </label>
        {running ? (
          <button className="ae-btn ae-btn--danger" onClick={() => api.stop()}>Stop after round</button>
        ) : (
          <button className="ae-btn ae-btn--primary" onClick={launch} disabled={launching}>
            {launching ? "Launching…" : hasRun ? "Launch new run" : "Launch agents"}
          </button>
        )}
      </div>

      {!hasRun ? (
        <div className="ae-card ae-card--padded" style={{ textAlign: "center", color: "var(--ae-text-muted)", padding: 40 }}>
          No runs yet for <span className="mono" style={{ color: "var(--ae-text-strong)" }}>{taskName}</span>. Launch the agents to begin.
        </div>
      ) : (
        <>
          <KpiStrip state={state} attempts={attempts} />
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2fr) minmax(300px, 1fr)", gap: 12, alignItems: "start" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
              <div className="ae-card">
                <div style={{ padding: "12px 16px 4px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span className="ae-stat__label">{metric} per experiment</span>
                  <span style={{ fontSize: 11, color: "var(--ae-text-dim)" }}>click a point to inspect</span>
                </div>
                <div style={{ padding: "4px 8px 8px" }}>
                  <ExperimentChart
                    attempts={[...attempts].sort((a, b) => a.created_at.localeCompare(b.created_at))}
                    baseline={baseline}
                    metric={metric}
                    selectedId={selected}
                    onSelect={setSelected}
                  />
                </div>
              </div>
              <div className="ae-card">
                <div style={{ padding: "12px 16px 4px" }}>
                  <span className="ae-stat__label">Experiments</span>
                </div>
                {run && (
                  <AttemptsTable task={taskName} run={run} attempts={attempts} metric={metric} selectedId={selected} onSelect={setSelected} />
                )}
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
              <LiveAgents attempts={attempts} onSelect={setSelected} />
              <NotesPanel notes={notes} />
            </div>
          </div>
        </>
      )}
    </StepShell>
  );
}
