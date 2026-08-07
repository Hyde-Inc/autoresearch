import React from "react";

export interface StepDef {
  key: string;
  label: string;
  sub: string;
}

interface Props {
  steps: StepDef[];
  active: number;
  furthest: number;
  onSelect: (index: number) => void;
}

export function Stepper({ steps, active, furthest, onSelect }: Props): React.ReactElement {
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: 0, width: "100%" }}>
      {steps.map((step, i) => {
        const done = i < furthest;
        const current = i === active;
        const reachable = i <= furthest;
        return (
          <React.Fragment key={step.key}>
            <button
              onClick={() => reachable && onSelect(i)}
              disabled={!reachable}
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "10px 12px",
                background: current ? "var(--ae-surface-hi)" : "transparent",
                border: "none",
                borderBottom: current
                  ? "2px solid var(--fk-blue)"
                  : "2px solid transparent",
                cursor: reachable ? "pointer" : "not-allowed",
                textAlign: "left",
                opacity: reachable ? 1 : 0.4,
                transition: "background 140ms ease",
              }}
            >
              <span
                style={{
                  flex: "0 0 auto",
                  width: 24,
                  height: 24,
                  borderRadius: 999,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  fontWeight: 700,
                  background: done
                    ? "var(--ae-success)"
                    : current
                      ? "var(--fk-blue)"
                      : "var(--ae-surface-elev)",
                  color: done || current ? "#fff" : "var(--ae-text-muted)",
                  border: done || current ? "none" : "1px solid var(--ae-divider-strong)",
                }}
              >
                {done ? "✓" : i + 1}
              </span>
              <span style={{ minWidth: 0 }}>
                <div
                  style={{
                    fontSize: 12.5,
                    fontWeight: 600,
                    color: current ? "var(--ae-text-strong)" : "var(--ae-text)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {step.label}
                </div>
                <div
                  style={{
                    fontSize: 10.5,
                    color: "var(--ae-text-dim)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {step.sub}
                </div>
              </span>
            </button>
            {i < steps.length - 1 && (
              <div style={{ width: 1, background: "var(--ae-divider)", flex: "0 0 auto" }} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
