import { useCallback, useEffect, useState } from "react";
import { AppHeader } from "./components/AppHeader";
import { AttemptsTable } from "./components/AttemptsTable";
import { DefineMetricModal } from "./components/DefineMetricModal";
import { ExperimentChart } from "./components/ExperimentChart";
import { KpiStrip } from "./components/KpiStrip";
import { LiveAgents } from "./components/LiveAgents";
import { NewRunModal } from "./components/NewRunModal";
import { NotesPanel } from "./components/NotesPanel";
import type { Attempt, Note, RunState, TaskRuns } from "./lib/api";
import { api } from "./lib/api";

const POLL_MS = 2500;

export default function App(): React.ReactElement {
  const [tasks, setTasks] = useState<TaskRuns[]>([]);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [state, setState] = useState<RunState>({});
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [selectedAttempt, setSelectedAttempt] = useState<string | null>(null);
  const [showNewRun, setShowNewRun] = useState(false);
  const [showDefineMetric, setShowDefineMetric] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const index = await api.runs();
      setTasks(index.tasks);
      let task = selectedTask;
      let run = selectedRun;
      if (!task || !run) {
        // Prefer a running run, else the latest with attempts, else the latest
        outer: for (const t of index.tasks) {
          for (const r of t.runs) {
            if (r.status === "running") {
              task = t.task;
              run = r.run;
              break outer;
            }
          }
        }
        if ((!task || !run) && index.tasks.length > 0) {
          const t = index.tasks[0];
          const withAttempts = t.runs.find((r) => r.attempts > 0);
          task = t.task;
          run = (withAttempts ?? t.runs[0]).run;
        }
        if (task && run) {
          setSelectedTask(task);
          setSelectedRun(run);
        }
      }
      if (task && run) {
        const [runState, runAttempts, runNotes] = await Promise.all([
          api.state(task, run),
          api.attempts(task, run),
          api.notes(task, run),
        ]);
        setState(runState);
        setAttempts(runAttempts);
        setNotes(runNotes);
      }
    } catch {
      // Server briefly unavailable; try again on the next tick.
    }
  }, [selectedTask, selectedRun]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, POLL_MS);
    return () => clearInterval(interval);
  }, [refresh]);

  const metric = state.primary_metric ?? "wmape";
  const baseline = state.baseline?.[metric];

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <AppHeader
        tasks={tasks}
        selectedTask={selectedTask}
        selectedRun={selectedRun}
        state={state}
        onSelect={(task, run) => {
          setSelectedTask(task);
          setSelectedRun(run);
          setSelectedAttempt(null);
        }}
        onNewRun={() => setShowNewRun(true)}
        onStop={() => api.stop()}
        onDefineMetric={() => setShowDefineMetric(true)}
      />

      <main style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column", gap: 12, maxWidth: 1400, width: "100%", margin: "0 auto" }}>
        {state.goal && (
          <div style={{ fontSize: 12.5, color: "var(--ae-text-muted)" }}>
            <span className="ae-stat__label" style={{ marginRight: 8 }}>Goal</span>
            {state.goal}
          </div>
        )}

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
                  selectedId={selectedAttempt}
                  onSelect={setSelectedAttempt}
                />
              </div>
            </div>

            <div className="ae-card">
              <div style={{ padding: "12px 16px 4px" }}>
                <span className="ae-stat__label">Experiments</span>
              </div>
              {selectedTask && selectedRun && (
                <AttemptsTable
                  task={selectedTask}
                  run={selectedRun}
                  attempts={attempts}
                  metric={metric}
                  selectedId={selectedAttempt}
                  onSelect={setSelectedAttempt}
                />
              )}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
            <LiveAgents attempts={attempts} onSelect={setSelectedAttempt} />
            <NotesPanel notes={notes} />
          </div>
        </div>
      </main>

      {showNewRun && <NewRunModal onClose={() => setShowNewRun(false)} onStarted={refresh} />}
      {showDefineMetric && <DefineMetricModal onClose={() => setShowDefineMetric(false)} />}
    </div>
  );
}
