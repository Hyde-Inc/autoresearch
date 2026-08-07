export interface Idea {
  title: string;
  hypothesis: string;
  instructions: string;
  category: string;
}

export interface Attempt {
  id: string;
  round: number;
  idea: Idea;
  status: "running" | "passed" | "rejected" | "failed";
  branch: string | null;
  commit: string | null;
  metrics: Record<string, number>;
  holdout_metrics: Record<string, number>;
  guardrail_failures: string[];
  error: string | null;
  promoted: boolean;
  created_at: string;
  duration_s: number | null;
}

export interface RunState {
  task?: string;
  status?: string;
  goal?: string;
  primary_metric?: string;
  metric_direction?: string;
  baseline?: Record<string, number>;
  incumbent?: Record<string, number>;
  round?: number;
  completed?: number;
}

export interface RunSummary {
  run: string;
  status: string;
  round: number;
  completed: number;
  attempts: number;
  goal: string;
  primary_metric: string;
}

export interface TaskRuns {
  task: string;
  runs: RunSummary[];
}

export interface Note {
  heading: string;
  body: string;
}

export interface LogTurn {
  type: "text" | "tool";
  text?: string;
  tool?: string;
  status?: string;
  summary?: string;
  error?: string | null;
  timestamp?: number;
}

export interface AgentLog {
  turns: LogTurn[];
  total_tokens: number;
  total_cost_usd: number;
}

export interface WorkedExample {
  rows: Record<string, string | number>[];
  steps: string[];
  value: number;
}

export interface MetricSpec {
  name: string;
  direction: "min" | "max";
  description: string;
  understanding: string;
  example: WorkedExample;
  code: string;
}

export interface MetricCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface MetricValidation {
  passed: boolean;
  checks: MetricCheck[];
  warnings: string[];
}

export interface MetricInfo {
  name: string;
  direction: string;
  custom: MetricSpec | null;
  columns: string[];
}

export interface InterpretResult {
  spec: MetricSpec;
  validation: MetricValidation;
  columns: string[];
}

export interface TaskInfo {
  config: string;
  name: string;
  description: string;
  goal: string;
  metric: string;
  custom_metric: boolean;
  bundled: boolean;
}

export interface IngestPreview {
  ok: boolean;
  columns: string[];
  rows: number;
  skus: number;
  date_min: string | null;
  date_max: string | null;
  distinct_dates: number;
  sample: Record<string, string | number>[];
  errors: string[];
  warnings: string[];
  suggested_validation_days: number;
  suggested_holdout_days: number;
}

export interface IngestResult {
  task: string;
  config: string;
  config_path: string;
  task_dir: string;
  train_rows: number;
  validation_rows: number;
  holdout_rows: number;
  skus: number;
  train_end: string;
  validation_end: string;
  holdout_end: string;
}

export const REQUIRED_SCHEMA = ["date", "sku_name", "sales", "selling_price"];

async function get<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `${url}: ${response.status}`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch {
      // keep the generic message
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  runs: () => get<{ tasks: TaskRuns[] }>("/api/runs"),
  state: (task: string, run: string) => get<RunState>(`/api/runs/${task}/${run}/state`),
  attempts: (task: string, run: string) => get<Attempt[]>(`/api/runs/${task}/${run}/attempts`),
  notes: (task: string, run: string) => get<Note[]>(`/api/runs/${task}/${run}/notes`),
  log: (task: string, run: string, id: string) =>
    get<AgentLog>(`/api/runs/${task}/${run}/attempts/${id}/log`),
  diff: async (task: string, run: string, id: string): Promise<string> => {
    const response = await fetch(`/api/runs/${task}/${run}/attempts/${id}/diff`);
    if (!response.ok) throw new Error("diff not found");
    return response.text();
  },
  start: (body: {
    config?: string;
    goal?: string;
    parallel?: number;
    max_experiments?: number;
  }) =>
    fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
  stop: () => fetch("/api/stop", { method: "POST" }).then((r) => r.json()),
  tasks: () => get<{ tasks: TaskInfo[] }>("/api/tasks"),
  report: async (task: string, run: string): Promise<string> => {
    const response = await fetch(`/api/runs/${task}/${run}/report`);
    if (!response.ok) throw new Error("report not found");
    return response.text();
  },
  ingestPreview: (file: File): Promise<IngestPreview> => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/ingest/preview", { method: "POST", body: form }).then((r) => r.json());
  },
  ingest: async (
    file: File,
    name: string,
    opts: { validation_days?: number; holdout_days?: number; overwrite?: boolean } = {},
  ): Promise<IngestResult> => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    if (opts.validation_days != null) form.append("validation_days", String(opts.validation_days));
    if (opts.holdout_days != null) form.append("holdout_days", String(opts.holdout_days));
    form.append("overwrite", String(opts.overwrite ?? false));
    const response = await fetch("/api/ingest", { method: "POST", body: form });
    if (!response.ok) {
      let detail = `ingest failed: ${response.status}`;
      try {
        const payload = await response.json();
        if (typeof payload.detail === "string") detail = payload.detail;
      } catch {
        // keep generic
      }
      throw new Error(detail);
    }
    return response.json();
  },
  metric: (config: string) => get<MetricInfo>(`/api/metric?config=${encodeURIComponent(config)}`),
  interpretMetric: (description: string, config: string) =>
    post<InterpretResult>("/api/metric/interpret", { description, config }),
  confirmMetric: (spec: MetricSpec, config: string) =>
    post<{ ok: boolean; definition: string; name: string; direction: string }>(
      "/api/metric/confirm",
      { spec, config },
    ),
  baseline: (config: string) =>
    get<{ code: string; path: string | null }>(`/api/baseline?config=${encodeURIComponent(config)}`),
  saveBaseline: (config: string, code: string) =>
    post<{ ok: boolean; path: string }>("/api/baseline", { config, code }),
};

export function fmtPct(value: number | undefined | null, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtNum(value: number | undefined | null, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export const STATUS_COLOR: Record<Attempt["status"], string> = {
  running: "var(--ae-warning)",
  passed: "var(--ae-success)",
  rejected: "var(--ae-danger)",
  failed: "#8893a8",
};

export const STATUS_PILL: Record<Attempt["status"], string> = {
  running: "ae-pill--amber",
  passed: "ae-pill--green",
  rejected: "ae-pill--red",
  failed: "ae-pill--gray",
};

const CATEGORY_PILLS = ["ae-pill--blue", "ae-pill--green", "ae-pill--yellow", "ae-pill--gray"];

export function categoryPill(category: string): string {
  let hash = 0;
  for (const ch of category) hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  return CATEGORY_PILLS[Math.abs(hash) % CATEGORY_PILLS.length];
}
