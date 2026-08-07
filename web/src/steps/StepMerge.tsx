import React, { useEffect, useState } from "react";
import type { Attempt } from "../lib/api";
import { api, fmtPct } from "../lib/api";
import { StepShell } from "./StepShell";

interface Props {
  task: string | null;
  run: string | null;
  onBack: () => void;
  onRestart: () => void;
}

function DiffView({ diff }: { diff: string }): React.ReactElement {
  return (
    <div className="ae-diff" style={{ maxHeight: 460, overflow: "auto", padding: "10px 12px" }}>
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

export function StepMerge({ task, run, onBack, onRestart }: Props): React.ReactElement {
  const [promoted, setPromoted] = useState<Attempt[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [diff, setDiff] = useState<string>("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!task || !run) return;
    api.attempts(task, run).then((all) => {
      const p = all.filter((a) => a.promoted);
      setPromoted(p);
      if (p.length && !selected) setSelected(p[p.length - 1].id);
    });
  }, [task, run]);

  useEffect(() => {
    if (!task || !run || !selected) return;
    setDiff("");
    api.diff(task, run, selected).then(setDiff).catch(() => setDiff("(no diff available)"));
  }, [task, run, selected]);

  const copyPatch = async (): Promise<void> => {
    await navigator.clipboard.writeText(diff);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const winner = promoted.find((a) => a.id === selected);

  return (
    <StepShell
      title="Merge the winning ideas"
      description={
        <>
          These experiments beat your baseline on the metric and passed every guardrail — including
          the hidden holdout. Review the exact code changes and merge them into your own codebase.
        </>
      }
      onBack={onBack}
      onNext={onRestart}
      nextLabel="Start a new project →"
      wide
    >
      {promoted.length === 0 ? (
        <div className="ae-card ae-card--padded" style={{ textAlign: "center", color: "var(--ae-text-muted)", padding: 40 }}>
          Nothing was promoted in this run, so there's nothing to merge. Try another run with a
          larger experiment budget or a refined goal.
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 12, alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {promoted.map((a) => (
              <button
                key={a.id}
                onClick={() => setSelected(a.id)}
                className="ae-card"
                style={{
                  textAlign: "left",
                  padding: "10px 12px",
                  cursor: "pointer",
                  border: selected === a.id ? "1px solid var(--fk-blue)" : "1px solid var(--ae-divider)",
                  background: selected === a.id ? "var(--ae-surface-hi)" : "var(--ae-surface)",
                }}
              >
                <div style={{ fontSize: 12.5, fontWeight: 600, color: "var(--ae-text-strong)", display: "flex", justifyContent: "space-between" }}>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.idea.title}</span>
                  <span style={{ color: "var(--ae-success)" }}>★</span>
                </div>
                <div className="mono" style={{ fontSize: 11, color: "var(--ae-text-muted)", marginTop: 3 }}>
                  {a.id} · holdout {fmtPct(a.holdout_metrics?.[a.metrics ? Object.keys(a.metrics)[0] : "wmape"])}
                </div>
              </button>
            ))}
          </div>

          <div className="ae-card" style={{ minWidth: 0 }}>
            <div style={{ padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--ae-divider)" }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ae-text-strong)" }}>{winner?.idea.title}</div>
                <div className="mono" style={{ fontSize: 11, color: "var(--ae-text-dim)" }}>solution/ changes · commit {winner?.commit?.slice(0, 8) ?? "—"}</div>
              </div>
              <button className="ae-btn" onClick={copyPatch} disabled={!diff}>
                {copied ? "Copied ✓" : "Copy patch"}
              </button>
            </div>
            {diff ? <DiffView diff={diff} /> : <div style={{ padding: 16, fontSize: 12, color: "var(--ae-text-muted)" }}>Loading diff…</div>}
            <div style={{ padding: "10px 14px", borderTop: "1px solid var(--ae-divider)", fontSize: 11.5, color: "var(--ae-text-muted)", lineHeight: 1.6 }}>
              To apply locally: save the patch to <span className="mono">winner.patch</span> and run{" "}
              <span className="mono">git apply winner.patch</span> from your project root, or copy the
              changed <span className="mono">solution/train.py</span> directly into your model repo.
            </div>
          </div>
        </div>
      )}
    </StepShell>
  );
}
