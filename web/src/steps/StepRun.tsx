import React, { useCallback, useEffect, useState } from "react";
import { AttemptsTable } from "../components/AttemptsTable";
import { ExperimentChart } from "../components/ExperimentChart";
import { KpiStrip } from "../components/KpiStrip";
import { LiveAgents } from "../components/LiveAgents";
import { NotesPanel } from "../components/NotesPanel";
import { SessionsSidebar } from "../components/SessionsSidebar";
import type { Attempt, Note, RunState } from "../lib/api";
import { api } from "../lib/api";
import { StepShell } from "./StepShell";

const POLL_MS = 2500;
const MAX_AGENTS = 4;
const MAX_ROUNDS = 4;

const TRACK_PRESETS = [
  "Statistical models (ARIMA / ETS / Theta)",
  "Gradient-boosted trees (XGBoost / LightGBM)",
  "Chronos / foundation forecasting models",
  "Feature engineering + price elasticity",
  "Ensembles & per-item bias calibration",
  "Open — any approach",
];

interface Props {
  config: string;
  taskName: string;
  run: string | null;
  onSelectRun: (run: string | null) => void;
  onBack: () => void;
  onDone: () => void;
}

export function StepRun({ config, taskName, run, onSelectRun, onBack, onDone }: Props): React.ReactElement {
  return (
    <div style={{ maxWidth: 1400, width: "100%", margin: "0 auto", display: "grid", gridTemplateColumns: "260px 1fr", gap: 14, alignItems: "start" }}>
      <SessionsSidebar taskName={taskName} selectedRun={run} onSelect={onSelectRun} onNew={() => onSelectRun(null)} />
      {run ? (
        <RunMonitor taskName={taskName} run={run} onBack={onBack} onDone={onDone} />
      ) : (
        <RunConfigForm config={config} taskName={taskName} onLaunched={onSelectRun} onBack={onBack} />
      )}
    </div>
  );
}

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }): React.ReactElement {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <span className="ae-stat__label">{label}</span>
      {children}
      {hint && <span style={{ fontSize: 11, color: "var(--ae-text-dim)" }}>{hint}</span>}
    </label>
  );
}

function RunConfigForm({
  config,
  taskName,
  onLaunched,
  onBack,
}: {
  config: string;
  taskName: string;
  onLaunched: (run: string) => void;
  onBack: () => void;
}): React.ReactElement {
  const [agents, setAgents] = useState(3);
  const [rounds, setRounds] = useState(2);
  const [roundMinutes, setRoundMinutes] = useState(15);
  const [costLimit, setCostLimit] = useState<number | "">(5);
  const [goal, setGoal] = useState("");
  const [tracks, setTracks] = useState<string[]>(["", "", ""]);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setAgentCount = (n: number): void => {
    const clamped = Math.min(MAX_AGENTS, Math.max(1, n));
    setAgents(clamped);
    setTracks((prev) => {
      const next = prev.slice(0, clamped);
      while (next.length < clamped) next.push("");
      return next;
    });
  };

  const launch = async (): Promise<void> => {
    setLaunching(true);
    setError(null);
    try {
      await api.start({
        config,
        goal: goal.trim() || undefined,
        parallel: agents,
        rounds,
        round_timeout_s: roundMinutes * 60,
        cost_limit_usd: costLimit === "" ? null : Number(costLimit),
        tracks: tracks.map((t) => t.trim()),
      });
      // Poll for the freshly created run and hand control to the monitor.
      const started = Date.now();
      const poll = async (): Promise<void> => {
        const index = await api.runs();
        const task = index.tasks.find((t) => t.task === taskName);
        const newest = task?.runs[0];
        if (newest && (newest.status === "running" || Date.now() - started > 20000)) {
          onLaunched(newest.run);
          return;
        }
        if (Date.now() - started < 20000) setTimeout(poll, 1200);
        else setLaunching(false);
      };
      setTimeout(poll, 1200);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setLaunching(false);
    }
  };

  const totalExperiments = agents * rounds;

  return (
    <StepShell
      title="Configure the Autoresearch run"
      description={
        <>
          Set the search budget and give each agent a research track. Once you launch, the
          configuration is locked and the run begins. <span style={{ color: "var(--ae-text-dim)" }}>Demo limits: ≤4 agents, ≤4 rounds.</span>
        </>
      }
      onBack={onBack}
    >
      <div className="ae-card ae-card--padded" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        <Field label="Agents" hint="parallel per round">
          <input className="ae-input" type="number" min={1} max={MAX_AGENTS} value={agents} onChange={(e) => setAgentCount(Number(e.target.value))} />
        </Field>
        <Field label="Rounds" hint="run sequentially">
          <input className="ae-input" type="number" min={1} max={MAX_ROUNDS} value={rounds} onChange={(e) => setRounds(Math.min(MAX_ROUNDS, Math.max(1, Number(e.target.value))))} />
        </Field>
        <Field label="Time / round (min)" hint="per-agent budget">
          <input className="ae-input" type="number" min={1} value={roundMinutes} onChange={(e) => setRoundMinutes(Math.max(1, Number(e.target.value)))} />
        </Field>
        <Field label="Cost limit (USD)" hint="blank = no cap">
          <input className="ae-input" type="number" min={0} step={0.5} value={costLimit} onChange={(e) => setCostLimit(e.target.value === "" ? "" : Number(e.target.value))} />
        </Field>
      </div>

      <Field label="Goal (optional override)">
        <input className="ae-input" value={goal} placeholder="Reduce error while keeping runtime and holdout stable" onChange={(e) => setGoal(e.target.value)} />
      </Field>

      <div className="ae-card ae-card--padded" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="ae-stat__label">Agent tracks</span>
          <span style={{ fontSize: 11.5, color: "var(--ae-text-dim)" }}>
            {totalExperiments} experiments total ({agents} agents × {rounds} rounds)
          </span>
        </div>
        {tracks.map((track, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="ae-pill ae-pill--blue" style={{ flex: "0 0 auto" }}>Agent {i + 1}</span>
              <input
                className="ae-input"
                style={{ flex: 1 }}
                placeholder="Focus for this agent (leave blank for open exploration)"
                value={track}
                onChange={(e) => setTracks((prev) => prev.map((t, j) => (j === i ? e.target.value : t)))}
              />
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {TRACK_PRESETS.map((preset) => (
                <button
                  key={preset}
                  onClick={() => setTracks((prev) => prev.map((t, j) => (j === i ? preset : t)))}
                  className="ae-pill ae-pill--gray"
                  style={{ cursor: "pointer", border: "none", fontSize: 10.5 }}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div style={{ border: "1px solid rgba(239,94,94,0.4)", background: "rgba(239,94,94,0.08)", borderRadius: "var(--ae-radius-sm)", padding: "10px 12px", fontSize: 12.5, color: "#ff8c8c" }}>
          {error}
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "flex-end", borderTop: "1px solid var(--ae-divider)", paddingTop: 14 }}>
        <button className="ae-btn ae-btn--primary" onClick={launch} disabled={launching}>
          {launching ? "Launching agents…" : "Lock config & launch run"}
        </button>
      </div>
    </StepShell>
  );
}

function fmtDuration(sec: number): string {
  if (sec % 60 === 0) return `${sec / 60} min`;
  return `${sec}s`;
}

function RunMonitor({
  taskName,
  run,
  onBack,
  onDone,
}: {
  taskName: string;
  run: string;
  onBack: () => void;
  onDone: () => void;
}): React.ReactElement {
  const [state, setState] = useState<RunState>({});
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, a, n] = await Promise.all([
        api.state(taskName, run),
        api.attempts(taskName, run),
        api.notes(taskName, run),
      ]);
      setState(s);
      setAttempts(a);
      setNotes(n);
    } catch {
      // transient
    }
  }, [taskName, run]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  const running = state.status === "running";
  const metric = state.primary_metric ?? "wmape";
  const baseline = state.baseline?.[metric];
  const cfg = state.run_config;

  return (
    <div className="ae-fade-in" style={{ display: "flex", flexDirection: "column", gap: 14, minWidth: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontFamily: "var(--ae-font-display)", fontSize: 20, fontWeight: 600, color: "var(--ae-text-strong)" }}>
            {running ? "Run in progress" : "Run complete"}
          </div>
          <div style={{ fontSize: 12.5, color: "var(--ae-text-muted)", marginTop: 2 }}>
            {taskName} · round {state.round ?? 0}/{cfg?.rounds ?? "?"} · {state.completed ?? 0} experiments
            {state.total_cost_usd != null ? ` · $${state.total_cost_usd.toFixed(2)} spent` : ""}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {running && (
            <button className="ae-btn ae-btn--danger" onClick={() => api.stop()}>Stop after round</button>
          )}
          <button className="ae-btn ae-btn--primary" onClick={onDone}>See learnings →</button>
        </div>
      </div>

      {/* Locked, read-only configuration */}
      {cfg && (
        <div className="ae-card ae-card--padded" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="ae-stat__label">Locked configuration</span>
            <span className="ae-pill ae-pill--gray" style={{ fontSize: 10 }}>🔒 read-only</span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, fontSize: 12.5 }}>
            <span><b>{cfg.agents}</b> agents</span>
            <span><b>{cfg.rounds}</b> rounds</span>
            <span><b>{fmtDuration(cfg.round_timeout_s)}</b> / round</span>
            <span>cost limit <b>{cfg.cost_limit_usd != null ? `$${cfg.cost_limit_usd}` : "none"}</b></span>
          </div>
          {cfg.tracks && cfg.tracks.some((t) => t) && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {cfg.tracks.map((t, i) => (
                <div key={i} style={{ fontSize: 12, display: "flex", gap: 8, alignItems: "baseline" }}>
                  <span className="ae-pill ae-pill--blue" style={{ fontSize: 10 }}>Agent {i + 1}</span>
                  <span style={{ color: t ? "var(--ae-text)" : "var(--ae-text-dim)" }}>{t || "open — any approach"}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <KpiStrip state={state} attempts={attempts} />

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2fr) minmax(280px, 1fr)", gap: 12, alignItems: "start" }}>
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
            <AttemptsTable task={taskName} run={run} attempts={attempts} metric={metric} selectedId={selected} onSelect={setSelected} />
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
          <LiveAgents attempts={attempts} onSelect={setSelected} />
          <NotesPanel notes={notes} />
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", borderTop: "1px solid var(--ae-divider)", paddingTop: 14 }}>
        <button className="ae-btn" onClick={onBack}>← Back</button>
        <button className="ae-btn ae-btn--primary" onClick={onDone}>See learnings →</button>
      </div>
    </div>
  );
}
