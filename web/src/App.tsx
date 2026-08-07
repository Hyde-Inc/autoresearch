import { useCallback, useEffect, useState } from "react";
import { Stepper, type StepDef } from "./components/Stepper";
import type { IngestResult, TaskInfo } from "./lib/api";
import { api } from "./lib/api";
import { StepBaseline } from "./steps/StepBaseline";
import { StepIngest } from "./steps/StepIngest";
import { StepLearnings } from "./steps/StepLearnings";
import { StepMerge } from "./steps/StepMerge";
import { StepMetric } from "./steps/StepMetric";
import { StepRun } from "./steps/StepRun";

const STEPS: StepDef[] = [
  { key: "ingest", label: "Ingest data", sub: "Upload sales CSV" },
  { key: "metric", label: "Define metric", sub: "Metric & grader" },
  { key: "baseline", label: "Baseline model", sub: "Starting point" },
  { key: "run", label: "Autoresearch run", sub: "Trigger & monitor" },
  { key: "learnings", label: "Learnings", sub: "What it found" },
  { key: "merge", label: "Merge ideas", sub: "Ship the wins" },
];

export default function App(): React.ReactElement {
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [config, setConfig] = useState<string | null>(null);
  const [taskName, setTaskName] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  const [furthest, setFurthest] = useState(0);
  const [run, setRun] = useState<string | null>(null);

  const loadTasks = useCallback(async () => {
    try {
      const { tasks } = await api.tasks();
      setTasks(tasks);
      if (!config && tasks.length > 0) {
        const preferred = tasks.find((t) => !t.bundled) ?? tasks[0];
        setConfig(preferred.config);
        setTaskName(preferred.name);
        setFurthest((f) => Math.max(f, STEPS.length - 1));
      }
    } catch {
      // server starting up
    }
  }, [config]);

  useEffect(() => {
    loadTasks();
  }, [loadTasks]);

  const goto = (index: number): void => {
    setActive(index);
    setFurthest((f) => Math.max(f, index));
  };

  const selectTask = (cfg: string): void => {
    const info = tasks.find((t) => t.config === cfg);
    setConfig(cfg);
    setTaskName(info?.name ?? null);
    setRun(null);
    setFurthest(STEPS.length - 1);
  };

  const onIngested = (result: IngestResult): void => {
    setConfig(result.config);
    setTaskName(result.task);
    setRun(null);
    loadTasks();
    goto(1);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 24,
          padding: "10px 20px",
          background: "var(--ae-surface)",
          borderBottom: "1px solid var(--ae-divider)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14, minWidth: 0 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: "linear-gradient(135deg, var(--fk-blue) 0%, #7c3aed 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: "var(--ae-font-display)",
              fontWeight: 700,
              fontSize: 15,
              color: "#fff",
            }}
          >
            A
          </div>
          <div>
            <div
              style={{
                fontFamily: "var(--ae-font-display)",
                fontSize: 15,
                fontWeight: 600,
                letterSpacing: "-0.02em",
                color: "var(--ae-text-strong)",
                lineHeight: 1.2,
              }}
            >
              Autoresearch Lab
            </div>
            <div style={{ fontSize: 11.5, color: "var(--ae-text-muted)" }}>
              From raw sales data to a shipped forecasting model
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="ae-stat__label">Project</span>
          <select
            className="ae-select"
            value={config ?? ""}
            onChange={(e) => selectTask(e.target.value)}
          >
            {config === null && <option value="">— none yet —</option>}
            {tasks.map((t) => (
              <option key={t.config} value={t.config}>
                {t.name}
                {t.bundled ? " (demo)" : ""}
              </option>
            ))}
          </select>
        </div>
      </header>

      <div style={{ background: "var(--ae-surface)", borderBottom: "1px solid var(--ae-divider)", padding: "0 12px" }}>
        <div style={{ maxWidth: 1400, margin: "0 auto" }}>
          <Stepper steps={STEPS} active={active} furthest={furthest} onSelect={goto} />
        </div>
      </div>

      <main style={{ flex: 1, padding: "22px 16px 40px", width: "100%" }}>
        {active === 0 && <StepIngest onIngested={onIngested} existingConfig={config} />}

        {active === 1 &&
          (config ? (
            <StepMetric config={config} onBack={() => goto(0)} onDone={() => goto(2)} />
          ) : (
            <NeedTask />
          ))}

        {active === 2 &&
          (config ? (
            <StepBaseline config={config} onBack={() => goto(1)} onDone={() => goto(3)} />
          ) : (
            <NeedTask />
          ))}

        {active === 3 &&
          (config && taskName ? (
            <StepRun
              config={config}
              taskName={taskName}
              onBack={() => goto(2)}
              onDone={() => goto(4)}
              onRun={(_t, r) => setRun(r)}
            />
          ) : (
            <NeedTask />
          ))}

        {active === 4 && (
          <StepLearnings task={taskName} run={run} onBack={() => goto(3)} onDone={() => goto(5)} />
        )}

        {active === 5 && (
          <StepMerge task={taskName} run={run} onBack={() => goto(4)} onRestart={() => goto(0)} />
        )}
      </main>
    </div>
  );
}

function NeedTask(): React.ReactElement {
  return (
    <div style={{ maxWidth: 900, margin: "0 auto", textAlign: "center", color: "var(--ae-text-muted)", padding: 60 }}>
      Ingest a dataset first (step 1) to unlock this step.
    </div>
  );
}
