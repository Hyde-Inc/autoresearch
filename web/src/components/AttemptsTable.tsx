import React, { useEffect, useState } from "react";
import type { AgentLog, Attempt } from "../lib/api";
import { STATUS_PILL, api, categoryPill, fmtNum, fmtPct } from "../lib/api";

interface Props {
  task: string;
  run: string;
  attempts: Attempt[];
  metric: string;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

type Tab = "reasoning" | "log" | "diff";

function DiffView({ diff }: { diff: string }): React.ReactElement {
  return (
    <div className="ae-diff" style={{ maxHeight: 420, overflow: "auto", padding: "10px 12px" }}>
      {diff.split("\n").map((line, i) => {
        let cls = "";
        if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("diff ")) cls = "meta";
        else if (line.startsWith("@@")) cls = "hunk";
        else if (line.startsWith("+")) cls = "add";
        else if (line.startsWith("-")) cls = "del";
        return (
          <span key={i} className={cls || undefined}>
            {line || " "}
            {"\n"}
          </span>
        );
      })}
    </div>
  );
}

function LogView({ log }: { log: AgentLog }): React.ReactElement {
  return (
    <div style={{ maxHeight: 420, overflow: "auto", padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", gap: 8 }}>
        <span className="ae-pill ae-pill--blue tabular">{log.total_tokens.toLocaleString()} tokens</span>
        <span className="ae-pill ae-pill--yellow tabular">${log.total_cost_usd.toFixed(3)} cost</span>
        <span className="ae-pill ae-pill--gray tabular">{log.turns.length} events</span>
      </div>
      {log.turns.map((turn, i) =>
        turn.type === "text" ? (
          <div
            key={i}
            style={{
              fontSize: 12.5,
              lineHeight: 1.55,
              color: "var(--ae-text)",
              background: "var(--ae-surface-elev)",
              border: "1px solid var(--ae-divider)",
              borderRadius: "var(--ae-radius-sm)",
              padding: "8px 11px",
            }}
          >
            {turn.text}
          </div>
        ) : (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, paddingLeft: 4 }}>
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: 999,
                flexShrink: 0,
                background: turn.status === "error" ? "var(--ae-danger)" : "var(--ae-text-dim)",
              }}
            />
            <span className="mono" style={{ fontSize: 11, color: "#8fb4ff", flexShrink: 0 }}>
              {turn.tool}
            </span>
            <span
              className="mono"
              style={{
                fontSize: 11,
                color: turn.status === "error" ? "#ff8c8c" : "var(--ae-text-muted)",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {turn.error ?? turn.summary}
            </span>
          </div>
        ),
      )}
    </div>
  );
}

function MetricsGrid({ attempt, metric }: { attempt: Attempt; metric: string }): React.ReactElement {
  const keys = Array.from(
    new Set([...Object.keys(attempt.metrics ?? {}), ...Object.keys(attempt.holdout_metrics ?? {})]),
  );
  return (
    <table className="ae-table" style={{ fontSize: 12.5 }}>
      <thead>
        <tr>
          <th>Metric</th>
          <th className="num">Validation</th>
          <th className="num">Hidden holdout</th>
        </tr>
      </thead>
      <tbody>
        {keys.map((key) => {
          const isPct = key === "wmape" || key === "mape";
          const fmt = isPct ? fmtPct : fmtNum;
          return (
            <tr key={key}>
              <td className="mono" style={{ color: key === metric ? "var(--ae-text-strong)" : undefined, fontWeight: key === metric ? 600 : 400 }}>
                {key}
              </td>
              <td className="num tabular">{fmt(attempt.metrics?.[key])}</td>
              <td className="num tabular">{fmt(attempt.holdout_metrics?.[key])}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function AttemptDetail({ task, run, attempt, metric }: { task: string; run: string; attempt: Attempt; metric: string }): React.ReactElement {
  const [tab, setTab] = useState<Tab>("reasoning");
  const [log, setLog] = useState<AgentLog | null>(null);
  const [diff, setDiff] = useState<string | null>(null);

  useEffect(() => {
    setLog(null);
    setDiff(null);
    setTab("reasoning");
  }, [attempt.id]);

  useEffect(() => {
    if (tab === "log" && log === null) {
      api.log(task, run, attempt.id).then(setLog).catch(() => setLog({ turns: [], total_tokens: 0, total_cost_usd: 0 }));
    }
    if (tab === "diff" && diff === null) {
      api.diff(task, run, attempt.id).then(setDiff).catch(() => setDiff("(no diff available)"));
    }
  }, [tab, attempt.id, task, run, log, diff]);

  // Live-refresh the log while the agent is still working
  useEffect(() => {
    if (tab !== "log" || attempt.status !== "running") return;
    const interval = setInterval(() => {
      api.log(task, run, attempt.id).then(setLog).catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, [tab, attempt.status, attempt.id, task, run]);

  return (
    <div className="ae-fade-in" style={{ background: "var(--ae-surface-elev)", borderRadius: "var(--ae-radius-sm)", border: "1px solid var(--ae-divider)", margin: "4px 8px 10px" }}>
      <div className="ae-tabs" style={{ padding: "0 8px" }}>
        {(["reasoning", "log", "diff"] as Tab[]).map((t) => (
          <button key={t} className={`ae-tab ${tab === t ? "ae-tab--active" : ""}`} onClick={() => setTab(t)}>
            {t === "reasoning" ? "Reasoning" : t === "log" ? "Agent log" : "Code diff"}
          </button>
        ))}
      </div>
      {tab === "reasoning" && (
        <div style={{ padding: "12px 14px", display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 16 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <div className="ae-stat__label" style={{ marginBottom: 4 }}>Hypothesis</div>
              <div style={{ fontSize: 12.5, lineHeight: 1.6 }}>{attempt.idea.hypothesis}</div>
            </div>
            <div>
              <div className="ae-stat__label" style={{ marginBottom: 4 }}>Instructions given to agent</div>
              <div style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ae-text-muted)", whiteSpace: "pre-wrap" }}>
                {attempt.idea.instructions}
              </div>
            </div>
            {attempt.guardrail_failures.length > 0 && (
              <div>
                <div className="ae-stat__label" style={{ marginBottom: 4, color: "#ff8c8c" }}>Guardrail failures</div>
                {attempt.guardrail_failures.map((failure, i) => (
                  <div key={i} className="mono" style={{ fontSize: 11.5, color: "#ff8c8c", padding: "3px 0" }}>
                    ✕ {failure}
                  </div>
                ))}
              </div>
            )}
            {attempt.error && (
              <div>
                <div className="ae-stat__label" style={{ marginBottom: 4, color: "#ff8c8c" }}>Error</div>
                <div className="mono" style={{ fontSize: 11.5, color: "#ff8c8c" }}>{attempt.error}</div>
              </div>
            )}
          </div>
          <div>
            <div className="ae-stat__label" style={{ marginBottom: 6 }}>Metrics</div>
            <MetricsGrid attempt={attempt} metric={metric} />
          </div>
        </div>
      )}
      {tab === "log" &&
        (log ? <LogView log={log} /> : <div style={{ padding: 16, fontSize: 12, color: "var(--ae-text-muted)" }}>Loading agent log…</div>)}
      {tab === "diff" &&
        (diff !== null ? <DiffView diff={diff} /> : <div style={{ padding: 16, fontSize: 12, color: "var(--ae-text-muted)" }}>Loading diff…</div>)}
    </div>
  );
}

export function AttemptsTable(props: Props): React.ReactElement {
  const { task, run, attempts, metric, selectedId, onSelect } = props;
  const ordered = [...attempts].sort((a, b) => a.created_at.localeCompare(b.created_at));
  return (
    <div style={{ overflow: "auto" }}>
      <table className="ae-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Round</th>
            <th>Experiment</th>
            <th>Category</th>
            <th>Status</th>
            <th className="num">Val {metric}</th>
            <th className="num">Holdout {metric}</th>
            <th className="num">RMSE</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((attempt) => (
            <React.Fragment key={attempt.id}>
              <tr
                className="clickable"
                onClick={() => onSelect(selectedId === attempt.id ? null : attempt.id)}
                style={selectedId === attempt.id ? { background: "rgba(40,116,240,0.08)" } : undefined}
              >
                <td className="mono" style={{ color: "var(--ae-text-muted)" }}>{attempt.id}</td>
                <td className="tabular">{attempt.round}</td>
                <td style={{ maxWidth: 340, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", color: "var(--ae-text-strong)", fontWeight: 500 }}>
                  {attempt.idea.title}
                </td>
                <td>
                  <span className={`ae-pill ${categoryPill(attempt.idea.category)}`}>{attempt.idea.category}</span>
                </td>
                <td>
                  <span className={`ae-pill ${STATUS_PILL[attempt.status]}`}>
                    {attempt.status === "running" && (
                      <span className="ae-pulse" style={{ width: 6, height: 6, borderRadius: 999, background: "var(--ae-warning)" }} />
                    )}
                    {attempt.status}
                    {attempt.promoted ? " ★" : ""}
                  </span>
                </td>
                <td className="num tabular">{fmtPct(attempt.metrics?.[metric])}</td>
                <td className="num tabular">{fmtPct(attempt.holdout_metrics?.[metric])}</td>
                <td className="num tabular">{fmtNum(attempt.metrics?.rmse)}</td>
                <td style={{ color: "var(--ae-text-dim)", fontSize: 11 }}>{selectedId === attempt.id ? "▾" : "▸"}</td>
              </tr>
              {selectedId === attempt.id && (
                <tr>
                  <td colSpan={9} style={{ padding: 0, background: "transparent" }}>
                    <AttemptDetail task={task} run={run} attempt={attempt} metric={metric} />
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
          {ordered.length === 0 && (
            <tr>
              <td colSpan={9} style={{ textAlign: "center", padding: 24, color: "var(--ae-text-muted)" }}>
                No experiments yet — start a run to see attempts appear here live.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
