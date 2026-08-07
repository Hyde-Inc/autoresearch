import React, { useEffect, useState } from "react";
import type { InterpretResult, MetricInfo } from "../lib/api";
import { api } from "../lib/api";

const THINKING_STEPS = [
  "Reading your metric definition…",
  "Restating the metric precisely…",
  "Working a small example by hand…",
  "Writing the grader code…",
  "Executing the code against the hand-worked example…",
];

const PLACEHOLDER =
  "e.g. Weighted MAPE, but under-forecasting hurts twice as much as over-forecasting " +
  "because stockouts lose customers. Penalize under-forecasts at 2x weight.";

function Thinking(): React.ReactElement {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setStep((s) => Math.min(s + 1, THINKING_STEPS.length - 1)), 3500);
    return () => clearInterval(t);
  }, []);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: "18px 4px" }}>
      {THINKING_STEPS.map((label, i) => (
        <div key={label} style={{ display: "flex", alignItems: "center", gap: 10, opacity: i <= step ? 1 : 0.3 }}>
          <span
            className={i === step ? "ae-pulse" : undefined}
            style={{
              width: 8,
              height: 8,
              borderRadius: 999,
              background: i < step ? "var(--ae-success)" : i === step ? "var(--fk-blue-hi)" : "var(--ae-divider-strong)",
              boxShadow: i === step ? "0 0 8px var(--fk-blue-hi)" : "none",
              flex: "0 0 auto",
            }}
          />
          <span style={{ fontSize: 12.5, color: i <= step ? "var(--ae-text)" : "var(--ae-text-dim)" }}>{label}</span>
        </div>
      ))}
    </div>
  );
}

export function DefineMetricModal({ onClose }: { onClose: () => void }): React.ReactElement {
  const [current, setCurrent] = useState<MetricInfo | null>(null);
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<InterpretResult | null>(null);
  const [adopted, setAdopted] = useState<{ name: string; direction: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.metric().then(setCurrent).catch(() => {});
  }, []);

  const interpret = async (): Promise<void> => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.interpretMetric(description));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const adopt = async (): Promise<void> => {
    if (!result) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await api.confirmMetric(result.spec);
      setAdopted({ name: saved.name, direction: saved.direction });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const spec = result?.spec;
  const columns = spec ? Object.keys(spec.example.rows[0] ?? {}) : [];

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
        style={{
          width: 680,
          maxHeight: "88vh",
          overflowY: "auto",
          padding: 20,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <div style={{ fontFamily: "var(--ae-font-display)", fontSize: 16, fontWeight: 600, color: "var(--ae-text-strong)" }}>
            Define the evaluation metric
          </div>
          <div style={{ fontSize: 12, color: "var(--ae-text-muted)", marginTop: 2 }}>
            Describe the metric in plain English. The agent restates it, proves it by hand on a
            tiny example, and writes the grader — the code is only accepted if it reproduces the
            hand calculation.
          </div>
          {current && (
            <div style={{ marginTop: 8, display: "flex", gap: 6, alignItems: "center" }}>
              <span className="ae-stat__label">Currently grading on</span>
              <span className="ae-pill ae-pill--blue">{current.name} · {current.direction}</span>
              <span className="ae-stat__label" style={{ marginLeft: 8 }}>Eval columns</span>
              <span className="mono" style={{ fontSize: 11.5, color: "var(--ae-text-muted)" }}>
                {current.columns.join(", ")}
              </span>
            </div>
          )}
        </div>

        {adopted ? (
          <div
            style={{
              border: "1px solid rgba(62, 194, 122, 0.4)",
              background: "rgba(62, 194, 122, 0.08)",
              borderRadius: "var(--ae-radius-sm)",
              padding: 16,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ fontWeight: 600, color: "#79d6a1" }}>Metric adopted as the grader</div>
            <div style={{ fontSize: 12.5 }}>
              Every experiment will now be scored on{" "}
              <span className="mono" style={{ color: "var(--ae-text-strong)" }}>{adopted.name}</span>{" "}
              ({adopted.direction === "min" ? "lower is better" : "higher is better"}). The research
              director and all agents are told this definition, and the protected evaluator runs
              exactly the verified code you just reviewed.
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 6 }}>
              <button className="ae-btn ae-btn--primary" onClick={onClose}>Done</button>
            </div>
          </div>
        ) : (
          <>
            <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <span className="ae-stat__label">Your metric, in your words</span>
              <textarea
                className="ae-textarea"
                rows={3}
                placeholder={PLACEHOLDER}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                style={{ resize: "vertical" }}
                disabled={busy}
              />
            </label>

            {error && (
              <div
                style={{
                  border: "1px solid rgba(239, 94, 94, 0.4)",
                  background: "rgba(239, 94, 94, 0.08)",
                  borderRadius: "var(--ae-radius-sm)",
                  padding: "10px 12px",
                  fontSize: 12.5,
                  color: "#ff8c8c",
                }}
              >
                {error}
              </div>
            )}

            {busy && !result && <Thinking />}

            {spec && result && (
              <div className="ae-fade-in" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="ae-pill ae-pill--blue mono">{spec.name}</span>
                  <span className={`ae-pill ${spec.direction === "min" ? "ae-pill--green" : "ae-pill--yellow"}`}>
                    {spec.direction === "min" ? "lower is better" : "higher is better"}
                  </span>
                </div>

                <div>
                  <div className="ae-stat__label" style={{ marginBottom: 4 }}>Agent's understanding</div>
                  <div style={{ fontSize: 13, lineHeight: 1.6 }}>{spec.understanding}</div>
                </div>

                <div>
                  <div className="ae-stat__label" style={{ marginBottom: 6 }}>Hand-worked example</div>
                  <div style={{ border: "1px solid var(--ae-divider)", borderRadius: "var(--ae-radius-sm)", overflow: "hidden" }}>
                    <table className="ae-table">
                      <thead>
                        <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
                      </thead>
                      <tbody>
                        {spec.example.rows.map((row, i) => (
                          <tr key={i}>
                            {columns.map((c) => (
                              <td key={c} className={typeof row[c] === "number" ? "num mono" : undefined}>
                                {String(row[c] ?? "")}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <ul style={{ margin: "10px 0 0", paddingLeft: 18, display: "flex", flexDirection: "column", gap: 4 }}>
                    {spec.example.steps.map((step, i) => (
                      <li key={i} style={{ fontSize: 12.5, color: "var(--ae-text-muted)" }}>{step}</li>
                    ))}
                  </ul>
                  <div style={{ marginTop: 8, fontSize: 13 }}>
                    <span className="ae-stat__label" style={{ marginRight: 8 }}>Hand-computed value</span>
                    <span className="mono" style={{ color: "var(--ae-text-strong)", fontWeight: 600 }}>
                      {spec.example.value}
                    </span>
                  </div>
                </div>

                <div>
                  <div className="ae-stat__label" style={{ marginBottom: 6 }}>Grader code (runs in the protected evaluator)</div>
                  <pre
                    className="mono"
                    style={{
                      margin: 0,
                      padding: 12,
                      fontSize: 11.5,
                      lineHeight: 1.55,
                      background: "var(--ae-surface-elev)",
                      border: "1px solid var(--ae-divider)",
                      borderRadius: "var(--ae-radius-sm)",
                      overflowX: "auto",
                      whiteSpace: "pre",
                    }}
                  >
                    {spec.code}
                  </pre>
                </div>

                <div>
                  <div className="ae-stat__label" style={{ marginBottom: 6 }}>Machine verification</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                    {result.validation.checks.map((check) => (
                      <div key={check.name} style={{ display: "flex", gap: 8, fontSize: 12.5, alignItems: "baseline" }}>
                        <span style={{ color: check.passed ? "var(--ae-success)" : "var(--ae-danger)", fontWeight: 700 }}>
                          {check.passed ? "✓" : "✗"}
                        </span>
                        <span className="mono" style={{ color: "var(--ae-text-strong)" }}>{check.name}</span>
                        <span style={{ color: "var(--ae-text-muted)" }}>{check.detail}</span>
                      </div>
                    ))}
                    {result.validation.warnings.map((warning, i) => (
                      <div key={i} style={{ display: "flex", gap: 8, fontSize: 12.5, alignItems: "baseline" }}>
                        <span style={{ color: "var(--ae-warning)", fontWeight: 700 }}>!</span>
                        <span style={{ color: "var(--ae-warning)" }}>{warning}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              <button className="ae-btn" onClick={onClose} disabled={busy}>Cancel</button>
              {spec ? (
                <>
                  <button className="ae-btn" onClick={interpret} disabled={busy || !description.trim()}>
                    {busy ? "Working…" : "Re-interpret"}
                  </button>
                  <button className="ae-btn ae-btn--primary" onClick={adopt} disabled={busy}>
                    {busy ? "Saving…" : "Adopt as grader"}
                  </button>
                </>
              ) : (
                <button
                  className="ae-btn ae-btn--primary"
                  onClick={interpret}
                  disabled={busy || !description.trim()}
                >
                  {busy ? "Working…" : "Interpret metric"}
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
