import React, { useRef, useState } from "react";
import type { IngestPreview, IngestResult } from "../lib/api";
import { REQUIRED_SCHEMA, api } from "../lib/api";
import { StepShell } from "./StepShell";

interface Props {
  onIngested: (result: IngestResult) => void;
  existingConfig: string | null;
}

export function StepIngest({ onIngested, existingConfig }: Props): React.ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [preview, setPreview] = useState<IngestPreview | null>(null);
  const [valDays, setValDays] = useState<number | "">("");
  const [holdDays, setHoldDays] = useState<number | "">("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadFile = async (f: File): Promise<void> => {
    setFile(f);
    setError(null);
    setPreview(null);
    if (!name) setName(f.name.replace(/\.csv$/i, ""));
    setBusy(true);
    try {
      const p = await api.ingestPreview(f);
      setPreview(p);
      if (p.suggested_validation_days) setValDays(p.suggested_validation_days);
      if (p.suggested_holdout_days) setHoldDays(p.suggested_holdout_days);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const create = async (): Promise<void> => {
    if (!file || !preview?.ok) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.ingest(file, name || file.name, {
        validation_days: valDays === "" ? undefined : Number(valDays),
        holdout_days: holdDays === "" ? undefined : Number(holdDays),
        overwrite: true,
      });
      onIngested(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const columns = preview?.sample?.[0] ? Object.keys(preview.sample[0]) : REQUIRED_SCHEMA;

  return (
    <StepShell
      title="Ingest your sales data"
      description={
        <>
          Upload a CSV with the fixed schema below. We validate it, clean it, and split it
          chronologically into training, validation, and a hidden holdout — the foundation every
          later step builds on.
        </>
      }
      onNext={preview?.ok ? create : undefined}
      nextLabel={busy ? "Creating task…" : "Create task & continue →"}
      nextDisabled={busy || !preview?.ok || !name.trim()}
      nextHint={
        existingConfig && !file ? "A task already exists — upload to replace, or skip ahead." : undefined
      }
    >
      <div className="ae-card ae-card--padded" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <span className="ae-stat__label">Required schema</span>
        {REQUIRED_SCHEMA.map((c) => (
          <span key={c} className="ae-pill ae-pill--blue mono">{c}</span>
        ))}
        <span style={{ fontSize: 11.5, color: "var(--ae-text-dim)", marginLeft: 4 }}>
          one row per item per day · extra columns are kept as features
        </span>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) loadFile(f);
        }}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `1.5px dashed ${dragging ? "var(--fk-blue)" : "var(--ae-divider-strong)"}`,
          background: dragging ? "rgba(40,116,240,0.06)" : "var(--ae-surface)",
          borderRadius: "var(--ae-radius-md)",
          padding: "28px 20px",
          textAlign: "center",
          cursor: "pointer",
          transition: "border-color 140ms ease, background 140ms ease",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) loadFile(f);
          }}
        />
        <div style={{ fontSize: 14, color: "var(--ae-text-strong)", fontWeight: 600 }}>
          {file ? file.name : "Drop a CSV here, or click to browse"}
        </div>
        <div style={{ fontSize: 12, color: "var(--ae-text-muted)", marginTop: 4 }}>
          {file
            ? `${(file.size / 1024).toFixed(0)} KB — click to choose a different file`
            : "We never leave your machine; the file is parsed locally by the dashboard server."}
        </div>
      </div>

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

      {preview && (
        <div className="ae-fade-in" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {preview.errors.map((e, i) => (
            <div key={i} style={{ fontSize: 12.5, color: "#ff8c8c" }}>✗ {e}</div>
          ))}

          {preview.ok && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                <Stat label="Rows" value={preview.rows.toLocaleString()} />
                <Stat label="Items (SKUs)" value={String(preview.skus)} />
                <Stat label="Distinct dates" value={String(preview.distinct_dates)} />
                <Stat label="Date range" value={`${preview.date_min} → ${preview.date_max}`} small />
              </div>

              {preview.warnings.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {preview.warnings.map((w, i) => (
                    <div key={i} style={{ fontSize: 12, color: "var(--ae-warning)" }}>! {w}</div>
                  ))}
                </div>
              )}

              <div>
                <div className="ae-stat__label" style={{ marginBottom: 6 }}>Preview (first rows)</div>
                <div style={{ border: "1px solid var(--ae-divider)", borderRadius: "var(--ae-radius-sm)", overflow: "auto", maxHeight: 220 }}>
                  <table className="ae-table">
                    <thead>
                      <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
                    </thead>
                    <tbody>
                      {preview.sample.map((row, i) => (
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
              </div>

              <div className="ae-card ae-card--padded" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr 1fr", gap: 14, alignItems: "end" }}>
                <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <span className="ae-stat__label">Task name</span>
                  <input className="ae-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Acme Retail" />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <span className="ae-stat__label">Validation days</span>
                  <input className="ae-input" type="number" min={1} value={valDays} onChange={(e) => setValDays(e.target.value === "" ? "" : Number(e.target.value))} />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <span className="ae-stat__label">Hidden holdout days</span>
                  <input className="ae-input" type="number" min={1} value={holdDays} onChange={(e) => setHoldDays(e.target.value === "" ? "" : Number(e.target.value))} />
                </label>
                <div style={{ gridColumn: "1 / -1", fontSize: 11.5, color: "var(--ae-text-dim)" }}>
                  The last {holdDays || "?"} days become a hidden holdout the agents never see; the
                  {" "}{valDays || "?"} days before that are the validation window used for scoring. Everything
                  earlier is training history.
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </StepShell>
  );
}

function Stat({ label, value, small }: { label: string; value: string; small?: boolean }): React.ReactElement {
  return (
    <div className="ae-card ae-card--padded" style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <span className="ae-stat__label">{label}</span>
      <span className="ae-stat__value" style={{ fontSize: small ? 14 : 24 }}>{value}</span>
    </div>
  );
}
