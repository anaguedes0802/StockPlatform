/*
 * StockPlatform front-end core: number formatting, the API client, the auth
 * store, live quotes over WebSocket and Lucide icons. Plain browser JavaScript,
 * no build step. Loaded on every page before Alpine.js starts.
 */

// ---------- formatting (locale pinned to en-US: $310.20, 1,234.56) ----------

const FMT_LOCALE = "en-US";

function fmtNumber(v, fractionDigits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString(FMT_LOCALE, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}

function fmtPct(v, fractionDigits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${Number(v).toFixed(fractionDigits)}%`;
}

function fmtCompact(v) {
  if (v == null) return "—";
  return new Intl.NumberFormat(FMT_LOCALE, { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

function changeColor(v) {
  if (v == null) return "text-muted";
  if (v > 0) return "text-success";
  if (v < 0) return "text-danger";
  return "text-muted";
}

/** Join class names, skipping falsy values (the `cn()` of the old React app). */
function cn(...parts) {
  return parts.flat().filter(Boolean).join(" ");
}

function relativeTime(iso) {
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** A plain (non-reactive) deep copy — hand this to chart libraries, never Alpine proxies. */
function plain(v) {
  return v == null ? v : JSON.parse(JSON.stringify(v));
}

// ---------- API client (same-origin /api proxy, see apps/web/app/main.py) ----------

const ACCESS_KEY = "sp.access";
const REFRESH_KEY = "sp.refresh";

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/** Human-readable message from an API error (FastAPI puts it in `detail`). */
function errMsg(e) {
  const raw = e && e.message ? e.message : String(e);
  try {
    const j = JSON.parse(raw);
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  } catch {}
  return raw;
}

async function request(path, init = {}) {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text || res.statusText);
  }
  if (res.status === 204) return null;
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

/** Same as request() but sends the signed-in user's access token. */
async function authedFetch(path, init = {}) {
  const token = localStorage.getItem(ACCESS_KEY);
  return request(path, {
    ...init,
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(init.headers || {}) },
  });
}

const post = (body) => ({ method: "POST", body: JSON.stringify(body) });

const api = {
  search: (q) => request(`/stocks/search?q=${encodeURIComponent(q)}`),
  detail: (symbol) => request(`/stocks/${symbol}`),
  history: (symbol, interval = "1d", range = "1y") =>
    request(`/stocks/${symbol}/history?interval=${interval}&range=${range}`),
  indicators: (symbol, names, interval = "1d", range = "1y") =>
    request(`/stocks/${symbol}/indicators?names=${names.join(",")}&interval=${interval}&range=${range}`),
  news: (symbol, limit = 10) => request(`/stocks/${symbol}/news?limit=${limit}`),
  priceAction: (symbol, interval = "1d", range = "1y") =>
    request(`/stocks/${symbol}/price-action?interval=${interval}&range=${range}`),
  insider: (symbol, limit = 20) => request(`/stocks/${symbol}/insider?limit=${limit}`),
  institutional: (symbol) => request(`/stocks/${symbol}/institutional`),
  politicians: (symbol, limit = 15) => request(`/stocks/${symbol}/politicians?limit=${limit}`),
  politiciansRecent: (limit = 30, days) =>
    request(`/politicians/recent?limit=${limit}${days ? `&days=${days}` : ""}`),
  politicianTrades: (name, limit = 200) => request(`/politicians/${encodeURIComponent(name)}?limit=${limit}`),
  congressTracker: () => request(`/politicians/tracker`),
  paperStatus: () => request(`/paper-strategy/status`),
  paperPlan: (refresh = false) => request(`/paper-strategy/plan${refresh ? "?refresh=true" : ""}`),
  paperReplay: () => request(`/paper-strategy/replay`),
  paperRebalance: (force = false) =>
    authedFetch(`/paper-strategy/rebalance${force ? "?force=true" : ""}`, { method: "POST" }),
  famousInvestors: () => request(`/famous-investors`),
  opinion: (symbol, useLlm, riskMode, timeframe) => {
    const params = new URLSearchParams();
    if (useLlm !== undefined) params.set("use_llm", String(useLlm));
    if (riskMode) params.set("risk_mode", riskMode);
    if (timeframe) params.set("timeframe", timeframe);
    const qs = params.toString();
    return request(`/ai/opinion/${symbol}${qs ? `?${qs}` : ""}`);
  },
  forecast: (symbol, horizons = ["1d", "5d", "30d"]) =>
    request(`/ai/forecast`, post({ symbol, horizons, models: ["ensemble"] })),
  recommend: (symbol) => request(`/ai/recommend`, post({ symbol })),
  trackRecord: () => request(`/track-record`),
  regime: () => request(`/regime`),
  botConfig: () => request(`/bot/config`),
  botBacktest: (body) => request(`/bot/backtest`, post(body)),
  botSignal: (symbol) => request(`/bot/signal/${symbol}`),
  swingConfig: () => request(`/swing/config`),
  swingScan: (body) => request(`/swing/scan`, post(body)),
  swingBacktest: (body) => request(`/swing/backtest`, post(body)),
  swingFilterEval: (body) => request(`/swing/filter-eval`, post(body)),
  gateReport: () => request(`/bot/gate/report`),
};

// ---------- auth store ----------

document.addEventListener("alpine:init", () => {
  Alpine.store("auth", {
    me: null,
    token: null,
    refresh: null,
    ready: false,

    async init() {
      this.token = localStorage.getItem(ACCESS_KEY);
      this.refresh = localStorage.getItem(REFRESH_KEY);
      await this.loadMe();
    },

    async loadMe() {
      if (!this.token) {
        this.me = null;
        this.ready = true;
        return;
      }
      try {
        this.me = await authedFetch("/auth/me");
      } catch {
        // access token expired: try the refresh token once, then give up
        if (this.refresh && (await this.tryRefresh())) return this.loadMe();
        this.clear();
      } finally {
        this.ready = true;
      }
    },

    async tryRefresh() {
      try {
        const r = await request("/auth/refresh", post({ refresh: this.refresh }));
        this.store(r);
        return true;
      } catch {
        return false;
      }
    },

    store(r) {
      localStorage.setItem(ACCESS_KEY, r.access);
      localStorage.setItem(REFRESH_KEY, r.refresh);
      this.token = r.access;
      this.refresh = r.refresh;
    },

    clear() {
      localStorage.removeItem(ACCESS_KEY);
      localStorage.removeItem(REFRESH_KEY);
      this.token = null;
      this.refresh = null;
      this.me = null;
    },

    async login(email, password) {
      this.store(await request("/auth/login", post({ email, password })));
      await this.loadMe();
    },

    async register(email, password, displayName) {
      this.store(await request("/auth/register", post({ email, password, display_name: displayName })));
      await this.loadMe();
    },

    logout() {
      this.clear();
    },
  });
});

/** Resolves once the auth store has finished loading (use before authed calls). */
function authReady() {
  return new Promise((resolve) => {
    const check = () => (Alpine.store("auth").ready ? resolve(Alpine.store("auth").me) : setTimeout(check, 50));
    check();
  });
}

// ---------- live quotes over WebSocket ----------

/**
 * Subscribe to live quotes for `symbols`; `onTick(quote)` fires per update.
 * Returns a function that closes the socket. WebSockets go straight to the API
 * (APP.apiPublicUrl), not through the /api proxy.
 */
function liveQuotes(symbols, onTick) {
  if (!symbols || symbols.length === 0) return () => {};
  const wsBase = (window.APP?.apiPublicUrl || "http://localhost:8000").replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/ws/quotes`);
  ws.onopen = () => ws.send(JSON.stringify({ action: "subscribe", symbols }));
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "quote" && msg.price) {
        onTick(msg);
      } else if (msg.type === "bar" && msg.c) {
        // Crypto (and US bars) arrive as 1-min bars: use the close as the live price.
        onTick({ symbol: msg.symbol, price: msg.c, change: null, change_pct: null, ts: msg.ts ?? null });
      }
    } catch {}
  };
  ws.onerror = () => {};
  const ping = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: "ping" }));
  }, 30_000);
  return () => {
    clearInterval(ping);
    if (ws.readyState === WebSocket.OPEN) ws.close();
  };
}

// ---------- Lucide icons ----------
// Write <i data-lucide="name" class="size-4"></i>; it is swapped for the SVG.
// Never put Alpine directives on the <i> itself (it gets replaced) — wrap it.

function toPascal(name) {
  return name.replace(/(^\w|-\w)/g, (s) => s.replace("-", "").toUpperCase());
}

function renderIcons(root = document) {
  if (!window.lucide) return;
  root.querySelectorAll("i[data-lucide]").forEach((el) => {
    const name = el.getAttribute("data-lucide");
    const node = lucide.icons[toPascal(name)];
    if (!node) return console.warn(`unknown icon: ${name}`);
    const [tag, attrs, children] = node;
    const svg = lucide.createElement([tag, { ...attrs, class: cn("lucide", `lucide-${name}`, el.getAttribute("class")) }, children]);
    for (const a of el.attributes) {
      if (a.name !== "class" && a.name !== "data-lucide") svg.setAttribute(a.name, a.value);
    }
    el.replaceWith(svg);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderIcons();
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      renderIcons();
    });
  }).observe(document.body, { childList: true, subtree: true });
});

// ---------- shared layout components ----------

function sidebar() {
  return {
    lastSymbol: "AAPL",
    init() {
      // "Stock analysis" reopens the last symbol you looked at instead of always AAPL.
      try {
        const m = location.pathname.match(/^\/stocks\/([^/]+)/);
        if (m) localStorage.setItem("sp:lastSymbol", decodeURIComponent(m[1]));
        this.lastSymbol = localStorage.getItem("sp:lastSymbol") || "AAPL";
      } catch {
        // storage unavailable (private mode): keep the default
      }
    },
  };
}

function stockSearch() {
  return {
    q: "",
    open: false,
    hits: [],
    timer: null,
    init() {
      this.$watch("q", () => this.lookup());
      this.lookup();
    },
    lookup() {
      clearTimeout(this.timer);
      this.timer = setTimeout(async () => {
        try {
          this.hits = await api.search(this.q);
        } catch {
          this.hits = [];
        }
      }, 150);
    },
    pick(symbol) {
      this.open = false;
      this.q = "";
      window.location.href = `/stocks/${symbol}`;
    },
  };
}

const SEV_DOT = { info: "bg-slate-400", warning: "bg-amber-400", critical: "bg-rose-500" };

function notificationsBell() {
  return {
    open: false,
    items: null,
    unread: 0,
    scanning: false,
    timer: null,
    async init() {
      const me = await authReady();
      if (!me) return;
      await this.refresh();
      // light polling every 60s for the badge: cheap GET, no LLM
      this.timer = setInterval(() => this.refresh(), 60_000);
    },
    async refresh() {
      if (!Alpine.store("auth").me) return;
      try {
        const list = await authedFetch("/notifications?limit=30");
        this.items = list;
        this.unread = list.filter((n) => !n.read_at).length;
      } catch {}
    },
    async scan() {
      this.scanning = true;
      try {
        await authedFetch("/notifications/scan", { method: "POST" });
        await this.refresh();
      } finally {
        this.scanning = false;
      }
    },
    async markRead(id) {
      try {
        await authedFetch(`/notifications/${id}/read`, { method: "POST" });
        const n = this.items?.find((x) => x.id === id);
        if (n && !n.read_at) {
          n.read_at = new Date().toISOString();
          this.unread = Math.max(0, this.unread - 1);
        }
      } catch {}
    },
    async markAllRead() {
      try {
        await authedFetch("/notifications/read-all", { method: "POST" });
        this.items?.forEach((n) => (n.read_at = n.read_at ?? new Date().toISOString()));
        this.unread = 0;
      } catch {}
    },
  };
}

// ---------- shared styling tables ----------

const VERDICT_STYLE = {
  STRONG_BUY: "bg-emerald-500/20 text-emerald-200 border-emerald-500/50",
  BUY: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  ACCUMULATE: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
  HOLD: "bg-slate-500/15 text-slate-200 border-slate-500/30",
  REDUCE: "bg-rose-500/10 text-rose-300 border-rose-500/30",
  SELL: "bg-rose-500/15 text-rose-300 border-rose-500/40",
  STRONG_SELL: "bg-rose-500/20 text-rose-200 border-rose-500/50",
};

const REGIME_STYLES = {
  risk_on: { cls: "border-emerald-500/40 bg-emerald-500/5", bar: "bg-emerald-400", label: "Risk-on" },
  neutral: { cls: "border-amber-500/40 bg-amber-500/5", bar: "bg-amber-400", label: "Neutral" },
  risk_off: { cls: "border-rose-500/40 bg-rose-500/5", bar: "bg-rose-400", label: "Risk-off" },
};

function regimeChipColor(score) {
  if (score >= 0.34) return "border-emerald-500/40 text-emerald-300";
  if (score <= -0.34) return "border-rose-500/40 text-rose-300";
  return "border-line text-muted";
}
