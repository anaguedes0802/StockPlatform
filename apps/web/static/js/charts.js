/*
 * Chart helpers on Chart.js (equity curves, comparisons, pies, bars) in the
 * platform's dark theme. Every helper takes a container element, builds a
 * <canvas> inside it and replaces any chart already there, so it is safe to
 * call again whenever the data changes. Data is copied with plain() first:
 * Chart.js must never hold Alpine's reactive proxies.
 */

const THEME = {
  accent: "#5EEAD4",
  muted: "#8B95A7",
  line: "#1E2533",
  surface2: "#161B26",
  text: "#E5E9F0",
  success: "#22C55E",
  danger: "#EF4444",
  warning: "#F59E0B",
  // distinct series colours (accent first), used by comparisons and pies
  palette: ["#5EEAD4", "#60A5FA", "#F59E0B", "#F472B6", "#A78BFA", "#34D399", "#FB923C", "#94A3B8", "#E879F9", "#FACC15"],
};

if (window.Chart) {
  Chart.defaults.color = THEME.muted;
  Chart.defaults.font.size = 11;
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.borderColor = THEME.line;
}

const TOOLTIP = {
  backgroundColor: THEME.surface2,
  borderColor: THEME.line,
  borderWidth: 1,
  titleColor: THEME.text,
  bodyColor: THEME.text,
  padding: 8,
  cornerRadius: 8,
};

/** Create (or recreate) a chart inside `el`. Returns the Chart instance. */
function mountChart(el, config, height) {
  if (!el) return null;
  if (el._chart) el._chart.destroy();
  el.innerHTML = "";
  if (height) el.style.height = `${height}px`;
  el.style.position = "relative";
  const canvas = document.createElement("canvas");
  el.appendChild(canvas);
  el._chart = new Chart(canvas, config);
  return el._chart;
}

function emptyState(el, height, text = "No data available") {
  if (el._chart) {
    el._chart.destroy();
    el._chart = null;
  }
  el.style.height = `${height}px`;
  el.innerHTML = `<div class="h-full grid place-items-center text-sm text-muted">${text}</div>`;
}

const lineScales = (yFormat) => ({
  x: { grid: { color: THEME.line }, border: { dash: [3, 3], color: THEME.line }, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
  y: {
    grid: { color: THEME.line },
    border: { dash: [3, 3], color: THEME.line },
    ticks: yFormat ? { callback: yFormat } : {},
  },
});

/**
 * Strategy vs benchmark equity curves (port of the old EquityChart).
 * strategy / benchmark: [{ts, equity}]. The benchmark is aligned by date.
 */
function equityChart(el, strategy, benchmark, labels = ["Strategy", "Buy & Hold"], height = 320) {
  strategy = plain(strategy) || [];
  benchmark = plain(benchmark) || [];
  if (strategy.length === 0) return emptyState(el, height);
  const byTs = new Map(benchmark.map((b) => [b.ts.slice(0, 10), b.equity]));
  const x = strategy.map((p) => p.ts.slice(0, 10));
  const bench = strategy.map((p, i) => byTs.get(p.ts.slice(0, 10)) ?? benchmark[i]?.equity ?? null);
  return mountChart(el, {
    type: "line",
    data: {
      labels: x,
      datasets: [
        { label: labels[0], data: strategy.map((p) => p.equity), borderColor: THEME.accent, borderWidth: 2, pointRadius: 0, tension: 0.2 },
        { label: labels[1], data: bench, borderColor: THEME.muted, borderWidth: 2, borderDash: [4, 4], pointRadius: 0, tension: 0.2, spanGaps: true },
      ],
    },
    options: {
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, color: THEME.muted } }, tooltip: TOOLTIP },
      scales: lineScales(),
    },
  }, height);
}

/**
 * Several named line series on a shared x axis.
 * series: [{label, data: [{x, y}], color?, dashed?}]
 */
function lineChart(el, series, { height = 320, yFormat, xFormat } = {}) {
  series = plain(series) || [];
  if (series.length === 0 || series.every((s) => !s.data?.length)) return emptyState(el, height);
  const xs = [...new Set(series.flatMap((s) => s.data.map((p) => p.x)))].sort();
  return mountChart(el, {
    type: "line",
    data: {
      labels: xFormat ? xs.map(xFormat) : xs,
      datasets: series.map((s, i) => {
        const byX = new Map(s.data.map((p) => [p.x, p.y]));
        return {
          label: s.label,
          data: xs.map((x) => byX.get(x) ?? null),
          borderColor: s.color || THEME.palette[i % THEME.palette.length],
          borderWidth: 2,
          borderDash: s.dashed ? [4, 4] : [],
          pointRadius: 0,
          tension: 0.2,
          spanGaps: true,
        };
      }),
    },
    options: {
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, color: THEME.muted } }, tooltip: TOOLTIP },
      scales: lineScales(yFormat),
    },
  }, height);
}

/** Doughnut chart. slices: [{name, value}] */
function pieChart(el, slices, { height = 260, valueFormat } = {}) {
  slices = (plain(slices) || []).filter((s) => s.value > 0);
  if (slices.length === 0) return emptyState(el, height);
  return mountChart(el, {
    type: "doughnut",
    data: {
      labels: slices.map((s) => s.name),
      datasets: [{
        data: slices.map((s) => s.value),
        backgroundColor: slices.map((_, i) => THEME.palette[i % THEME.palette.length]),
        borderColor: "#11151D",
        borderWidth: 2,
      }],
    },
    options: {
      maintainAspectRatio: false,
      animation: false,
      cutout: "55%",
      plugins: {
        legend: { position: "right", labels: { boxWidth: 10, color: THEME.muted } },
        tooltip: { ...TOOLTIP, callbacks: valueFormat ? { label: (c) => `${c.label}: ${valueFormat(c.raw)}` } : {} },
      },
    },
  }, height);
}

/**
 * Vertical bars. bars: [{label, value, color?}] or several datasets via
 * {labels, datasets: [{label, data, color}]}.
 */
function barChart(el, bars, { height = 260, yFormat, horizontal = false } = {}) {
  bars = plain(bars);
  let data;
  if (Array.isArray(bars)) {
    if (bars.length === 0) return emptyState(el, height);
    data = {
      labels: bars.map((b) => b.label),
      datasets: [{
        data: bars.map((b) => b.value),
        backgroundColor: bars.map((b) => b.color || (b.value >= 0 ? THEME.accent : THEME.danger)),
        borderRadius: 3,
      }],
    };
  } else {
    if (!bars?.labels?.length) return emptyState(el, height);
    data = {
      labels: bars.labels,
      datasets: bars.datasets.map((d, i) => ({
        label: d.label,
        data: d.data,
        backgroundColor: d.color || THEME.palette[i % THEME.palette.length],
        borderRadius: 3,
      })),
    };
  }
  const valueAxis = { grid: { color: THEME.line }, border: { dash: [3, 3], color: THEME.line }, ticks: yFormat ? { callback: yFormat } : {} };
  const catAxis = { grid: { display: false } };
  return mountChart(el, {
    type: "bar",
    data,
    options: {
      maintainAspectRatio: false,
      animation: false,
      indexAxis: horizontal ? "y" : "x",
      plugins: {
        legend: { display: !Array.isArray(bars), position: "bottom", labels: { boxWidth: 12, color: THEME.muted } },
        tooltip: TOOLTIP,
      },
      scales: horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
    },
  }, height);
}
