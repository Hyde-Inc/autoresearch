import React, { useMemo, useState } from "react";
import type { Attempt } from "../lib/api";
import { STATUS_COLOR, fmtPct } from "../lib/api";

interface Props {
  attempts: Attempt[];
  baseline: number | undefined;
  metric: string;
  selectedId: string | null;
  onSelect: (id: string) => void;
  height?: number;
}

const PAD = { top: 18, right: 16, bottom: 30, left: 46 };
const W = 760;

export function ExperimentChart(props: Props): React.ReactElement {
  const { attempts, baseline, metric, selectedId, onSelect, height = 240 } = props;
  const H = height;
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const points = useMemo(
    () =>
      attempts.map((a, i) => ({
        attempt: a,
        index: i,
        validation: a.metrics?.[metric] ?? null,
        holdout: a.holdout_metrics?.[metric] ?? null,
      })),
    [attempts, metric],
  );

  const yMax = useMemo(() => {
    let m = baseline ?? 0;
    for (const p of points) {
      if (p.validation != null) m = Math.max(m, p.validation);
      if (p.holdout != null) m = Math.max(m, p.holdout);
    }
    return (m || 0.2) * 1.15;
  }, [points, baseline]);

  const n = Math.max(points.length, 1);
  const xOf = (i: number): number =>
    PAD.left + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const yOf = (v: number): number => PAD.top + innerH - (v / yMax) * innerH;

  // Running-best (incumbent) trajectory, starting from the baseline
  const incumbentLine = useMemo(() => {
    if (baseline == null) return "";
    let best = baseline;
    const coords = [`${PAD.left},${yOf(best)}`];
    for (const p of points) {
      if (p.attempt.promoted && p.validation != null && p.validation < best) best = p.validation;
      coords.push(`${xOf(p.index)},${yOf(best)}`);
    }
    coords.push(`${PAD.left + innerW},${yOf(best)}`);
    return coords.join(" ");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, baseline, yMax]);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => yMax * f);
  const hover = hoverIdx != null ? points[hoverIdx] : null;

  const onMove = (e: React.MouseEvent<SVGSVGElement>): void => {
    if (points.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let bestI = 0;
    let bestD = Infinity;
    for (const p of points) {
      const d = Math.abs(xOf(p.index) - px);
      if (d < bestD) {
        bestD = d;
        bestI = p.index;
      }
    }
    setHoverIdx(bestI);
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      onMouseMove={onMove}
      onMouseLeave={() => setHoverIdx(null)}
      onClick={() => hover && onSelect(hover.attempt.id)}
      style={{ display: "block", cursor: hover ? "pointer" : "default" }}
    >
      {/* y grid + labels */}
      {yTicks.map((v) => (
        <g key={v}>
          <line x1={PAD.left} y1={yOf(v)} x2={W - PAD.right} y2={yOf(v)} stroke="rgba(255,255,255,0.05)" />
          <text x={PAD.left - 6} y={yOf(v) + 3} fontSize={9} fill="var(--ae-text-dim)" textAnchor="end">
            {(v * 100).toFixed(1)}%
          </text>
        </g>
      ))}

      {/* baseline */}
      {baseline != null && (
        <g>
          <line
            x1={PAD.left}
            y1={yOf(baseline)}
            x2={W - PAD.right}
            y2={yOf(baseline)}
            stroke="rgba(253,216,53,0.5)"
            strokeDasharray="4 3"
          />
          <text x={W - PAD.right - 4} y={yOf(baseline) - 5} fontSize={9} fill="rgba(255,216,98,0.9)" textAnchor="end">
            baseline {fmtPct(baseline, 1)}
          </text>
        </g>
      )}

      {/* incumbent trajectory */}
      {incumbentLine && (
        <polyline points={incumbentLine} fill="none" stroke="var(--fk-blue-hi)" strokeWidth={1.6} />
      )}

      {/* x labels */}
      {points.map((p) => (
        <text
          key={p.attempt.id}
          x={xOf(p.index)}
          y={H - 8}
          fontSize={9}
          fill={p.attempt.id === selectedId ? "var(--ae-text)" : "var(--ae-text-dim)"}
          textAnchor="middle"
          className="mono"
        >
          {p.attempt.id.slice(0, 6)}
        </text>
      ))}

      {/* holdout markers (hollow diamonds) */}
      {points.map(
        (p) =>
          p.holdout != null && (
            <g key={`h-${p.attempt.id}`} transform={`translate(${xOf(p.index)}, ${yOf(p.holdout)})`}>
              <rect
                x={-3.5}
                y={-3.5}
                width={7}
                height={7}
                transform="rotate(45)"
                fill="var(--ae-bg)"
                stroke="var(--ae-text-muted)"
                strokeWidth={1.2}
              />
            </g>
          ),
      )}

      {/* validation dots */}
      {points.map(
        (p) =>
          p.validation != null && (
            <g key={`v-${p.attempt.id}`}>
              {p.attempt.promoted && (
                <circle
                  cx={xOf(p.index)}
                  cy={yOf(p.validation)}
                  r={9}
                  fill="none"
                  stroke="rgba(74,140,255,0.5)"
                  strokeWidth={1.5}
                />
              )}
              <circle
                cx={xOf(p.index)}
                cy={yOf(p.validation)}
                r={p.attempt.id === selectedId ? 6 : 4.5}
                fill={STATUS_COLOR[p.attempt.status]}
                stroke="var(--ae-bg)"
                strokeWidth={1.5}
              />
            </g>
          ),
      )}

      {/* hover tooltip */}
      {hover && (
        <g>
          <line
            x1={xOf(hover.index)}
            y1={PAD.top}
            x2={xOf(hover.index)}
            y2={PAD.top + innerH}
            stroke="rgba(255,255,255,0.15)"
          />
          <g transform={`translate(${Math.min(xOf(hover.index) + 10, W - 230)}, ${PAD.top + 2})`}>
            <rect width={220} height={62} rx={5} fill="rgba(13,17,23,0.96)" stroke="var(--ae-divider)" />
            <text x={8} y={15} fontSize={10} fill="var(--ae-text-muted)">
              {hover.attempt.id} · round {hover.attempt.round} · {hover.attempt.status}
              {hover.attempt.promoted ? " · promoted" : ""}
            </text>
            <text x={8} y={31} fontSize={11} fontWeight={700} fill="var(--ae-text-strong)">
              {hover.attempt.idea.title.slice(0, 34)}
              {hover.attempt.idea.title.length > 34 ? "…" : ""}
            </text>
            <text x={8} y={48} fontSize={10} fill="var(--ae-text-muted)" className="tabular">
              val {fmtPct(hover.validation)} · holdout {fmtPct(hover.holdout)}
            </text>
          </g>
        </g>
      )}

      {/* axes */}
      <line x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} stroke="var(--ae-divider-strong)" />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} stroke="var(--ae-divider-strong)" />

      {/* legend */}
      <g transform={`translate(${PAD.left + 4}, ${PAD.top - 6})`}>
        <circle cx={4} cy={4} r={4} fill="var(--ae-success)" />
        <text x={12} y={7} fontSize={9} fill="var(--ae-text-muted)">passed</text>
        <circle cx={58} cy={4} r={4} fill="var(--ae-danger)" />
        <text x={66} y={7} fontSize={9} fill="var(--ae-text-muted)">rejected</text>
        <rect x={116} y={0} width={7} height={7} transform="rotate(45 119.5 3.5)" fill="none" stroke="var(--ae-text-muted)" strokeWidth={1.2} />
        <text x={130} y={7} fontSize={9} fill="var(--ae-text-muted)">hidden holdout</text>
        <line x1={202} y1={4} x2={220} y2={4} stroke="var(--fk-blue-hi)" strokeWidth={2} />
        <text x={224} y={7} fontSize={9} fill="var(--ae-text-muted)">incumbent</text>
        <line x1={286} y1={4} x2={304} y2={4} stroke="rgba(253,216,53,0.6)" strokeWidth={2} strokeDasharray="4 3" />
        <text x={308} y={7} fontSize={9} fill="var(--ae-text-muted)">baseline</text>
      </g>
    </svg>
  );
}
