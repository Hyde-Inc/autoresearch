import React from "react";
import type { Attempt, RunState } from "../lib/api";
import { fmtPct } from "../lib/api";

interface CardProps {
  label: string;
  value: string;
  sub?: string;
  deltaText?: string;
  deltaGood?: boolean | null;
  accent?: string;
}

function KpiCard(props: CardProps): React.ReactElement {
  const { label, value, sub, deltaText, deltaGood, accent } = props;
  const deltaColor =
    deltaGood == null ? "var(--ae-text-muted)" : deltaGood ? "var(--ae-success)" : "var(--ae-danger)";
  return (
    <div
      className="ae-card"
      style={{ padding: "13px 15px", display: "flex", flexDirection: "column", gap: 7, minWidth: 0 }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 2,
            background: accent ?? "var(--ae-text-dim)",
            flexShrink: 0,
          }}
        />
        <span className="ae-stat__label">{label}</span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
        <span className="ae-stat__value tabular" style={{ fontSize: 26 }}>
          {value}
        </span>
        {deltaText && (
          <span className="tabular" style={{ fontSize: 12.5, fontWeight: 600, color: deltaColor }}>
            {deltaText}
          </span>
        )}
      </div>
      {sub && <div className="ae-stat__sub tabular">{sub}</div>}
    </div>
  );
}

export function KpiStrip({
  state,
  attempts,
}: {
  state: RunState;
  attempts: Attempt[];
}): React.ReactElement {
  const metric = state.primary_metric ?? "wmape";
  const baseline = state.baseline?.[metric];
  const incumbent = state.incumbent?.[metric];
  const improvement =
    baseline != null && incumbent != null && baseline > 0 ? (baseline - incumbent) / baseline : null;
  const runningCount = attempts.filter((a) => a.status === "running").length;
  const passedCount = attempts.filter((a) => a.status === "passed").length;
  const promotedCount = attempts.filter((a) => a.promoted).length;
  const holdoutIncumbent = state.incumbent?.[`holdout_${metric}`];

  return (
    <div
      className="ae-fade-in"
      style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}
    >
      <KpiCard
        label={`Baseline ${metric}`}
        value={fmtPct(baseline)}
        accent="var(--ae-text-muted)"
        sub="Seasonal-naive seed"
      />
      <KpiCard
        label={`Best ${metric}`}
        value={fmtPct(incumbent)}
        accent="var(--fk-blue-hi)"
        deltaText={improvement != null ? `${improvement > 0 ? "−" : "+"}${Math.abs(improvement * 100).toFixed(1)}% vs base` : undefined}
        deltaGood={improvement != null ? improvement > 0 : null}
        sub={holdoutIncumbent != null ? `Holdout ${fmtPct(holdoutIncumbent)}` : undefined}
      />
      <KpiCard
        label="Round"
        value={String(state.round ?? 0)}
        accent="var(--fk-yellow)"
        sub={`${state.completed ?? 0} experiments completed`}
      />
      <KpiCard
        label="Experiments"
        value={`${passedCount}/${attempts.length}`}
        accent="var(--ae-success)"
        sub={`${promotedCount} promoted · ${attempts.length - passedCount - runningCount} rejected/failed`}
      />
      <KpiCard
        label="Agents live"
        value={String(runningCount)}
        accent={runningCount > 0 ? "var(--ae-warning)" : "var(--ae-text-dim)"}
        sub={runningCount > 0 ? "experiments in flight" : "idle"}
      />
    </div>
  );
}
