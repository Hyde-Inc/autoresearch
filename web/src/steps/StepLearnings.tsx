import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { SessionsSidebar } from "../components/SessionsSidebar";
import type { Attempt, Note, RunState } from "../lib/api";
import { api, fmtNum, fmtPct } from "../lib/api";
import { StepShell } from "./StepShell";

interface Props {
  task: string | null;
  run: string | null;
  onSelectRun: (run: string) => void;
  onBack: () => void;
  onDone: () => void;
}

export function StepLearnings({ task, run, onSelectRun, onBack, onDone }: Props): React.ReactElement {
  const [report, setReport] = useState<string>("");
  const [notes, setNotes] = useState<Note[]>([]);
  const [state, setState] = useState<RunState>({});
  const [attempts, setAttempts] = useState<Attempt[]>([]);

  useEffect(() => {
    if (!task || !run) return;
    api.report(task, run).then(setReport).catch(() => setReport(""));
    api.notes(task, run).then(setNotes).catch(() => {});
    api.state(task, run).then(setState).catch(() => {});
    api.attempts(task, run).then(setAttempts).catch(() => {});
  }, [task, run]);

  const metric = state.primary_metric ?? "wmape";
  const baseline = state.baseline?.[metric];
  const best = state.incumbent?.[metric];
  const improvement =
    baseline != null && best != null && baseline > 0 ? (baseline - best) / baseline : null;
  const promoted = attempts.filter((a) => a.promoted);

  const withSidebar = (content: React.ReactNode): React.ReactElement => (
    <div style={{ maxWidth: 1400, width: "100%", margin: "0 auto", display: "grid", gridTemplateColumns: "260px 1fr", gap: 14, alignItems: "start" }}>
      <SessionsSidebar taskName={task} selectedRun={run} onSelect={onSelectRun} />
      <div style={{ minWidth: 0 }}>{content}</div>
    </div>
  );

  if (!task || !run) {
    return withSidebar(
      <StepShell title="Learnings from the run" description="Pick a session on the left, or run the agents to generate findings." onBack={onBack}>
        <div className="ae-card ae-card--padded" style={{ textAlign: "center", color: "var(--ae-text-muted)", padding: 40 }}>
          No session selected yet — the director's findings and the improvement summary will appear here.
        </div>
      </StepShell>,
    );
  }

  return withSidebar(
    <StepShell
      title="Learnings from the run"
      description={
        <>
          What the lab discovered: how much the metric improved, which experiments won, and the
          research director's round-by-round reasoning about what worked and why.
        </>
      }
      onBack={onBack}
      onNext={onDone}
      nextLabel="Review & merge →"
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
        <Stat label={`Baseline ${metric}`} value={fmtPct(baseline)} />
        <Stat label={`Best ${metric}`} value={fmtPct(best)} accent="var(--fk-blue-hi)" />
        <Stat
          label="Improvement"
          value={improvement != null ? `${(improvement * 100).toFixed(1)}%` : "—"}
          accent={improvement && improvement > 0 ? "var(--ae-success)" : "var(--ae-text-dim)"}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 12, alignItems: "start" }}>
        <div className="ae-card" style={{ padding: "14px 18px" }}>
          <div className="ae-stat__label" style={{ marginBottom: 8 }}>Research report</div>
          <div className="ae-note-body" style={{ maxHeight: 460, overflow: "auto" }}>
            {report ? <ReactMarkdown>{report}</ReactMarkdown> : <span style={{ color: "var(--ae-text-muted)" }}>No report yet.</span>}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="ae-card" style={{ padding: "14px 18px" }}>
            <div className="ae-stat__label" style={{ marginBottom: 8 }}>Promoted experiments</div>
            {promoted.length === 0 ? (
              <div style={{ fontSize: 12.5, color: "var(--ae-text-muted)" }}>
                No experiment beat the baseline while passing every guardrail.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {promoted.map((a) => (
                  <div key={a.id} style={{ borderLeft: "2px solid var(--ae-success)", paddingLeft: 10 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ae-text-strong)" }}>
                      {a.idea.title} <span className="mono" style={{ color: "var(--ae-text-dim)", fontWeight: 400 }}>({a.id})</span>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--ae-text-muted)", marginTop: 2 }}>{a.idea.hypothesis}</div>
                    <div className="mono" style={{ fontSize: 11.5, marginTop: 4 }}>
                      {metric} {fmtPct(a.metrics?.[metric])} · holdout {fmtPct(a.holdout_metrics?.[metric])} · rmse {fmtNum(a.metrics?.rmse)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="ae-card" style={{ padding: "14px 18px" }}>
            <div className="ae-stat__label" style={{ marginBottom: 8 }}>Director reflections</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 260, overflow: "auto" }}>
              {notes.length === 0 && <div style={{ fontSize: 12, color: "var(--ae-text-muted)" }}>No reflections recorded.</div>}
              {notes.map((n, i) => (
                <div key={i}>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: "var(--ae-text-strong)" }}>{n.heading}</div>
                  <div className="ae-note-body" style={{ fontSize: 12 }}>
                    <ReactMarkdown>{n.body}</ReactMarkdown>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </StepShell>,
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }): React.ReactElement {
  return (
    <div className="ae-card ae-card--padded" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span className="ae-stat__label">{label}</span>
      <span className="ae-stat__value tabular" style={{ color: accent }}>{value}</span>
    </div>
  );
}
