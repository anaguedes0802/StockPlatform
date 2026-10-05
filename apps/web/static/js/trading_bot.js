/*
 * /trading-bot page logic (port of apps/web/app/trading-bot/page.tsx).
 * Backtests, the user's bots (trader + investor), the brokerage panel, the AI
 * gate track record, A/B experiments and the long-term DCA backtest.
 * Loaded from templates/pages/trading_bot.html via {% block head %}.
 */

/** Whole-dollar currency (the page-local fmtUsd of the original). */
function fmtUsd(n) {
  if (n == null) return "—";
  return Number(n).toLocaleString(FMT_LOCALE, {
    style: "currency", currency: "USD", minimumFractionDigits: 0, maximumFractionDigits: 0,
  });
}

const TB_TIMING_BADGE = {
  STRONG_BUY: { cls: "bg-success/20 text-success", label: "Strong buy" },
  ACCUMULATE: { cls: "bg-accent/20 text-accent", label: "Accumulate" },
  NORMAL: { cls: "bg-surface2 text-muted", label: "Normal" },
};

const TB_GATE_STYLE = {
  collecting: "border-l-line", helping: "border-l-success", hurting: "border-l-danger", inconclusive: "border-l-warning",
};

const TB_DCA_ROWS = [
  { key: "dca", label: "Diversified DCA" },
  { key: "dca_dip_tilt", label: "DCA + dip-tilt" },
  { key: "benchmark_spy", label: "100% SPY (benchmark)" },
];

/**
 * Controlled numeric input: clamp like the original onChange
 * (`Math.max(min, Number(v) || min)`) and write the value back to the field
 * when it differs, as React's controlled <input type="number"> does.
 */
function tbClampInput(ev, min) {
  const v = Math.max(min, Number(ev.target.value) || min);
  // eslint-disable-next-line eqeqeq
  if (v != ev.target.value) ev.target.value = v;
  return v;
}

function tradingBotPage() {
  return {
    config: null,

    // backtest controls
    mode: "portfolio",
    symbol: "SPY",
    busy: false,
    err: null,
    portfolio: null,
    single: null,

    // saved bots
    bots: null,
    scanning: null,

    // live execution
    account: null,
    running: null,
    lastRun: {},

    // investor bots
    previews: {},
    investRun: {},
    marketTiming: null,

    // DCA backtest
    dca: null,
    dcaMonthly: 1000,
    dcaBusy: false,

    // AI gate track record
    gate: null,
    gateErr: null,

    // A/B experiments
    exps: null,

    async init() {
      api.botConfig().then((c) => (this.config = c)).catch(() => {});
      // run a default portfolio backtest on first load so the win rate is visible
      void this.runBacktest();
      api.gateReport().then((r) => (this.gate = r)).catch((e) => (this.gateErr = errMsg(e)));

      const me = await authReady();
      this.onSignedIn(me);
      // Alpine's $watch fires on every dependency touch, so compare values
      // ourselves (React effects only re-ran when the dependency changed).
      let lastUser = me?.id ?? null;
      this.$watch("$store.auth.me", (m) => {
        if ((m?.id ?? null) === lastUser) return;
        lastUser = m?.id ?? null;
        this.onSignedIn(m);
      });
      // ExperimentPanel refetches whenever a bot's last run changes
      let lastKey = this.expKey;
      this.$watch("expKey", (k) => {
        if (k === lastKey) return;
        lastKey = k;
        this.loadExperiments();
      });
    },

    onSignedIn(me) {
      if (!me) return;
      void this.refreshBots();
      void this.refreshAccount();
      // investor: fetch the market-wide "good time to invest?" read once on load
      authedFetch("/bot/invest/timing").then((t) => (this.marketTiming = t)).catch(() => {});
      this.loadExperiments();
    },

    get me() {
      return Alpine.store("auth").me;
    },

    get expKey() {
      return this.bots?.map((b) => b.last_run_at).join("|") ?? "";
    },

    loadExperiments() {
      if (!this.me) return;
      authedFetch("/bot/experiments").then((e) => (this.exps = e)).catch(() => (this.exps = []));
    },

    async refreshBots() {
      if (!this.me) return;
      try { this.bots = await authedFetch("/bot/bots"); } catch {}
    },

    async refreshAccount() {
      if (!this.me) return;
      try { this.account = await authedFetch("/bot/account"); } catch {}
    },

    replaceBot(id, updated) {
      this.bots = this.bots?.map((x) => (x.id === id ? updated : x)) ?? null;
    },

    async runBacktest() {
      this.busy = true; this.err = null; this.portfolio = null; this.single = null;
      try {
        const body = this.mode === "single" ? { symbol: this.symbol.toUpperCase() } : {};
        const r = await api.botBacktest(body);
        if (r.mode === "single") this.single = r;
        else this.portfolio = r;
      } catch (e) { this.err = errMsg(e); }
      finally { this.busy = false; }
    },

    async createBot(kind = "trader") {
      if (!this.me) return;
      const body =
        kind === "picker"
          ? { name: "My Stock-Picker", bot_type: "investor", dsl: { pick_mode: "quality" } }
          : kind === "investor"
          ? { name: "My Investor", bot_type: "investor" }
          : { name: "My Bot", bot_type: "trader" };
      try { await authedFetch("/bot/bots", { method: "POST", body: JSON.stringify(body) }); void this.refreshBots(); }
      catch (e) { this.err = errMsg(e); }
    },

    async toggleBot(b) {
      const next = b.status === "active" ? "paused" : "active";
      await authedFetch(`/bot/bots/${b.id}/status/${next}`, { method: "POST" });
      void this.refreshBots();
    },

    async scanBot(b) {
      this.scanning = b.id;
      try {
        const updated = await authedFetch(`/bot/bots/${b.id}/scan`, { method: "POST" });
        this.replaceBot(b.id, updated);
      } finally { this.scanning = null; }
    },

    async deleteBot(b) {
      await authedFetch(`/bot/bots/${b.id}`, { method: "DELETE" });
      void this.refreshBots();
    },

    async setArmed(b, armed) {
      try {
        const updated = await authedFetch(`/bot/bots/${b.id}/${armed ? "arm" : "disarm"}`, { method: "POST" });
        this.replaceBot(b.id, updated);
      } catch (e) { this.err = errMsg(e); }
    },

    async toggleGate(b) {
      const next = !(b.dsl?.llm_gate ?? true);
      try {
        const updated = await authedFetch(`/bot/bots/${b.id}`, {
          method: "PATCH", body: JSON.stringify({ dsl: { ...b.dsl, llm_gate: next } }),
        });
        this.replaceBot(b.id, updated);
      } catch (e) { this.err = errMsg(e); }
    },

    async toggleFilter(b) {
      const next = !(b.dsl?.ml_filter ?? false);
      try {
        const updated = await authedFetch(`/bot/bots/${b.id}`, {
          method: "PATCH", body: JSON.stringify({ dsl: { ...b.dsl, ml_filter: next } }),
        });
        this.replaceBot(b.id, updated);
      } catch (e) { this.err = errMsg(e); }
    },

    async runLive(b) {
      this.running = b.id; this.err = null;
      try {
        const r = await authedFetch(`/bot/bots/${b.id}/run`, { method: "POST" });
        this.lastRun[b.id] = r;
        this.replaceBot(b.id, r.bot);
        void this.refreshAccount();
      } catch (e) { this.err = errMsg(e); }
      finally { this.running = null; }
    },

    async runDcaBacktest() {
      this.dcaBusy = true; this.err = null;
      try {
        this.dca = await authedFetch("/bot/invest/backtest", {
          method: "POST", body: JSON.stringify({ monthly_contribution: this.dcaMonthly }),
        });
      } catch (e) { this.err = errMsg(e); }
      finally { this.dcaBusy = false; }
    },

    async previewInvest(b, contribution) {
      this.err = null;
      try {
        const r = await authedFetch(`/bot/bots/${b.id}/rebalance-preview`, {
          method: "POST", body: JSON.stringify({ contribution }),
        });
        this.previews[b.id] = r;
        this.marketTiming = r.timing;
      } catch (e) { this.err = errMsg(e); }
    },

    async runInvest(b, contribution) {
      this.running = b.id; this.err = null;
      try {
        const r = await authedFetch(`/bot/bots/${b.id}/invest`, {
          method: "POST", body: JSON.stringify({ contribution }),
        });
        this.investRun[b.id] = r;
        this.replaceBot(b.id, r.bot);
        void this.refreshAccount();
      } catch (e) { this.err = errMsg(e); }
      finally { this.running = null; }
    },

    // ---------- view helpers ----------

    get strategyLine() {
      const s = this.config?.default_strategy;
      if (!s) return "";
      return `RSI(${s.rsi_period}) < ${s.rsi_entry} · TP ${(s.take_profit_pct * 100).toFixed(1)}% · exit SMA${s.exit_sma} · trend SMA${s.trend_sma}`;
    },

    get brokerReady() {
      return !!this.config?.broker?.configured;
    },
    get brokerPaper() {
      return this.account?.paper ?? this.config?.broker?.paper ?? true;
    },
    get brokerConfigured() {
      return this.account?.configured ?? this.config?.broker?.configured ?? false;
    },

    timingBadge(t) {
      return TB_TIMING_BADGE[t?.level] ?? TB_TIMING_BADGE.NORMAL;
    },

    isInvestor(b) {
      return b.bot_type === "investor";
    },
    gateOn(b) {
      return b.dsl?.llm_gate ?? true;
    },
    isSwing(b) {
      return String(b.dsl?.kind ?? "").startsWith("swing_");
    },
    armDisabled(b) {
      return !this.brokerReady && b.execution?.broker !== "sim";
    },
    armTitle(b) {
      return this.brokerReady || b.execution?.broker === "sim" ? "" : "Add Alpaca credentials first";
    },
    buysOf(b) {
      return (b.last_signals ?? []).filter((s) => s.action === "BUY");
    },
    actionCls(a) {
      return a.action === "BUY" ? "text-success"
        : a.action === "SELL" ? "text-accent"
        : a.action === "VETOED" || a.action === "ML_SKIPPED" ? "text-warning"
        : a.action.includes("FAILED") ? "text-danger"
        : "text-muted";
    },
    llmCls(d) {
      return d === "VETO" ? "bg-warning/20 text-warning"
        : d === "DOWNSIZE" ? "bg-accent/20 text-accent"
        : "bg-success/20 text-success";
    },
    swingTracked(b) {
      return Object.entries(b.run_state?.positions ?? {});
    },
    simTail(sim, autorun) {
      return `${sim.closed_trades} closed trade${sim.closed_trades === 1 ? "" : "s"}`
        + (sim.win_rate_pct != null ? ` · ${sim.win_rate_pct}% winners` : "")
        + (autorun ? " · runs daily at 09:45 New York time" : " · manual runs only");
    },

    // investor bots
    invTiming(b) {
      return this.investRun[b.id]?.timing ?? this.previews[b.id]?.timing;
    },
    invPlan(b) {
      return this.investRun[b.id]?.plan ?? this.previews[b.id]?.plan;
    },
    isPicker(b) {
      return b.dsl?.pick_mode === "quality";
    },
    picks(b) {
      return this.invPlan(b)?.picks ?? null;
    },
    holdsLabel(b) {
      if (!this.isPicker(b)) return "Target allocation";
      const p = this.picks(b);
      return `Quality stock picks ${p ? `(top ${p.length})` : ""}`;
    },
    allocEntries(b) {
      return Object.entries(b.dsl?.allocation ?? {});
    },
    allocPct(b, w) {
      const totalW = Object.values(b.dsl?.allocation ?? {}).reduce((a, v) => a + Number(v), 0) || 1;
      return Math.round((Number(w) / totalW) * 100);
    },
    planLine(plan) {
      return `Deploy ${fmtUsd(plan.deployable)} (${fmtUsd(plan.idle_cash)} idle cash${plan.contribution > 0 ? ` + ${fmtUsd(plan.contribution)} new` : ""})`;
    },
    investBuys(run) {
      return run.actions.filter((a) => a.action === "BUY").length;
    },

    // AI gate track record
    gateCounts(r) {
      return `${r.counts.judged} judged · ${r.counts.scored} scored · ${r.counts.pending} waiting for the trade to close`
        + (r.counts.fail_open ? ` · ${r.counts.fail_open} when the AI was offline (excluded)` : "");
    },
    gateOutcome(r, k) {
      const o = r.outcomes[k];
      return o.n ? `${o.win_rate_pct}% winners · ${fmtNumber(o.mean_r, 2)}R avg` : "no scored trades yet";
    },
    gateValueText(r) {
      const gv = r.gate_value;
      const value = gv.n
        ? `${gv.total_r > 0 ? "+" : ""}${fmtNumber(gv.total_r, 2)}R over ${gv.n} cut trades`
          + (gv.lo != null ? ` (per cut: ${fmtNumber(gv.lo, 2)} to ${fmtNumber(gv.hi, 2)}R, 90%)` : "")
        : "nothing to measure yet";
      return `Gate value so far: ${value}. Positive means the trades it vetoed or shrank went on to lose. ${r.method}`;
    },
    decisionCls(d) {
      return d === "VETO" ? "text-warning" : d === "DOWNSIZE" ? "text-accent" : "text-success";
    },

    // experiments
    expDays(e) {
      const days = e.arms[0]?.sim?.equity_history.length ?? 0;
      return `started ${e.started_at?.slice(0, 10) ?? "—"} · ${days} trading day${days === 1 ? "" : "s"} recorded`;
    },
    expNote(e) {
      const trades = Math.min(...e.arms.map((x) => x.sim?.closed_trades ?? 0));
      return trades < 100
        ? `${trades} closed trades per arm so far. Differences under ~100 trades are noise; judge after 6–12 months.`
        : "Enough trades to start comparing. Look for a gap that persists, not a lucky month.";
    },
    expHasChart(e) {
      const [a, b] = e.arms;
      return !!(a?.sim && b?.sim && a.sim.equity_history.length > 1);
    },
    drawExpChart(el, e) {
      const [a, b] = e.arms;
      if (!(a?.sim && b?.sim && a.sim.equity_history.length > 1)) return;
      equityChart(
        el,
        b.sim.equity_history.map((h) => ({ ts: h.date, equity: h.equity })),
        a.sim.equity_history.map((h) => ({ ts: h.date, equity: h.equity })),
        [b.arm ?? "B", a.arm ?? "A"],
      );
    },
  };
}
