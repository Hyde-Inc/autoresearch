import React, { useEffect, useState } from "react";
import { api } from "../lib/api";
import { StepShell } from "./StepShell";

interface Props {
  config: string;
  onBack: () => void;
  onDone: () => void;
}

export function StepBaseline({ config, onBack, onDone }: Props): React.ReactElement {
  const [code, setCode] = useState("");
  const [path, setPath] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoaded(false);
    api
      .baseline(config)
      .then((b) => {
        setCode(b.code);
        setPath(b.path);
        setLoaded(true);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [config]);

  const save = async (): Promise<void> => {
    setSaving(true);
    setError(null);
    try {
      const res = await api.saveBaseline(config, code);
      setPath(res.path);
      setDirty(false);
      setSavedAt(new Date().toLocaleTimeString());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const proceed = async (): Promise<void> => {
    if (dirty) await save();
    onDone();
  };

  return (
    <StepShell
      title="Set the baseline model"
      description={
        <>
          This is <code>solution/train.py</code> — the starting point every agent forks from and
          must beat. We seeded a schema-aware seasonal-naive baseline (repeat the same weekday from
          the most recent week). Edit it here, or paste your own current production model.
        </>
      }
      onBack={onBack}
      onNext={proceed}
      nextLabel={dirty ? "Save & continue →" : "Continue →"}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 11.5, color: "var(--ae-text-muted)" }}>
          {path ?? "solution/train.py"}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {savedAt && !dirty && (
            <span style={{ fontSize: 11.5, color: "var(--ae-success)" }}>saved {savedAt}</span>
          )}
          {dirty && <span style={{ fontSize: 11.5, color: "var(--ae-warning)" }}>unsaved changes</span>}
          <button className="ae-btn" onClick={save} disabled={saving || !dirty}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ fontSize: 12.5, color: "#ff8c8c" }}>{error}</div>
      )}

      <textarea
        className="ae-textarea mono"
        spellCheck={false}
        value={loaded ? code : "Loading…"}
        disabled={!loaded}
        onChange={(e) => {
          setCode(e.target.value);
          setDirty(true);
        }}
        style={{
          width: "100%",
          minHeight: 420,
          fontSize: 12,
          lineHeight: 1.55,
          resize: "vertical",
          whiteSpace: "pre",
          overflowWrap: "normal",
          overflowX: "auto",
          tabSize: 4,
        }}
      />

      <div className="ae-card ae-card--padded" style={{ fontSize: 12, color: "var(--ae-text-muted)", lineHeight: 1.6 }}>
        <strong style={{ color: "var(--ae-text-strong)" }}>Runtime contract:</strong> the evaluator
        sets <span className="mono">AUTORESEARCH_TRAIN_DATA</span>,{" "}
        <span className="mono">AUTORESEARCH_REQUEST</span>, and{" "}
        <span className="mono">AUTORESEARCH_OUTPUT</span> environment variables. Read the training
        parquet, forecast every requested row, and write a parquet with the id column, date column,
        and a <span className="mono">forecast</span> column. Forecasts must be finite and
        non-negative.
      </div>
    </StepShell>
  );
}
