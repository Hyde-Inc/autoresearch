import React from "react";

interface Props {
  title: string;
  description: React.ReactNode;
  children: React.ReactNode;
  onBack?: () => void;
  onNext?: () => void;
  nextLabel?: string;
  nextDisabled?: boolean;
  nextHint?: string;
  wide?: boolean;
}

export function StepShell(props: Props): React.ReactElement {
  const { title, description, children, onBack, onNext, nextLabel, nextDisabled, nextHint, wide } =
    props;
  return (
    <div
      className="ae-fade-in"
      style={{
        maxWidth: wide ? 1400 : 900,
        width: "100%",
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div>
        <div
          style={{
            fontFamily: "var(--ae-font-display)",
            fontSize: 20,
            fontWeight: 600,
            letterSpacing: "-0.02em",
            color: "var(--ae-text-strong)",
          }}
        >
          {title}
        </div>
        <div style={{ fontSize: 13, color: "var(--ae-text-muted)", marginTop: 3, maxWidth: 760 }}>
          {description}
        </div>
      </div>

      {children}

      {(onBack || onNext) && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderTop: "1px solid var(--ae-divider)",
            paddingTop: 14,
            marginTop: 4,
          }}
        >
          <div>
            {onBack && (
              <button className="ae-btn" onClick={onBack}>
                ← Back
              </button>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {nextHint && (
              <span style={{ fontSize: 12, color: "var(--ae-text-dim)" }}>{nextHint}</span>
            )}
            {onNext && (
              <button
                className="ae-btn ae-btn--primary"
                onClick={onNext}
                disabled={nextDisabled}
              >
                {nextLabel ?? "Continue →"}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
