/*
 * Candlestick chart: port of the old components/chart/CandlestickChart.tsx on
 * TradingView Lightweight Charts v5 (global `LightweightCharts`, loaded by base.html).
 *
 *   const cc = createCandlestickChart(el, { height: 460 });   // also kept on el._candleChart
 *   cc.setCandles([{time, open, high, low, close}]);         // time = UNIX seconds
 *   cc.setVolume([{time, value, color}]);
 *   cc.setOverlays(overlays, bands, markers);                // SMA/EMA lines, SMC zones, BOS/CHOCH markers
 *   cc.fitIfPending(visibleBars);                            // initial visible window (once per dataset)
 *   cc.requestFit();                                         // re-apply the window on the next fitIfPending()
 *   cc.setRulerMode(true | false);                           // measurement ruler
 *   cc.destroy();
 *
 * The controller holds the chart instance: keep it on the DOM element, never in
 * Alpine state, and pass it plain (non-proxy) data.
 */
(function () {
  const LC = window.LightweightCharts;

  // lightweight-charts renders the time axis in UTC by default. Each bar's `time`
  // is a real UNIX-epoch second, so `new Date(time*1000)` is the right instant and
  // toLocale* formats it in the browser's local timezone (axis + crosshair).
  function fmtAxisTick(time, tickMarkType, locale) {
    const d = new Date(time * 1000);
    switch (tickMarkType) {
      case LC.TickMarkType.Year:
        return String(d.getFullYear());
      case LC.TickMarkType.Month:
        return d.toLocaleDateString(locale, { month: "short", year: "2-digit" });
      case LC.TickMarkType.DayOfMonth:
        return d.toLocaleDateString(locale, { day: "numeric", month: "short" });
      case LC.TickMarkType.TimeWithSeconds:
        return d.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      case LC.TickMarkType.Time:
      default:
        return d.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit" });
    }
  }

  function fmtCrosshairTime(time) {
    return new Date(time * 1000).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  }

  // Keep cents (min 2) but reveal finer detail when it exists (premarket 227.345,
  // sub-dollar 0.8421); trailing zeros are trimmed. Axis, last-value label, crosshair.
  function fmtPrice(price) {
    return Number(price).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    });
  }

  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs) {
    const n = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    return n;
  }

  function createCandlestickChart(el, { height = 460 } = {}) {
    if (el._candleChart) el._candleChart.destroy();

    // <div class="relative" style="height"> <div container class="w-full h-full"/> ruler bits </div>
    el.innerHTML = "";
    el.classList.add("relative");
    el.style.height = `${height}px`;
    const container = document.createElement("div");
    container.className = "w-full h-full";
    el.appendChild(container);

    const chart = LC.createChart(container, {
      layout: { background: { color: "#0B0E14" }, textColor: "#8B95A7", attributionLogo: false },
      grid: { vertLines: { color: "#1E2533" }, horzLines: { color: "#1E2533" } },
      rightPriceScale: { borderColor: "#1E2533" },
      timeScale: {
        borderColor: "#1E2533", timeVisible: true, secondsVisible: false,
        // Axis labels in the user's local timezone (default is UTC).
        tickMarkFormatter: fmtAxisTick,
      },
      // Crosshair tooltip time in local timezone + higher-precision prices.
      localization: {
        timeFormatter: (t) => fmtCrosshairTime(t),
        priceFormatter: fmtPrice,
      },
      crosshair: { mode: LC.CrosshairMode.Magnet },
      width: container.clientWidth,
      height: container.clientHeight,
    });

    // Keep the chart sized to its container (window resize, sidebar toggle, layout
    // change, being shown after x-show). More robust than `autoSize`.
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      if (!e) return;
      const { width, height: h } = e.contentRect;
      if (width > 0 && h > 0) chart.resize(Math.floor(width), Math.floor(h));
    });
    ro.observe(container);

    const candleSeries = chart.addSeries(LC.CandlestickSeries, {
      upColor: "#22C55E", downColor: "#EF4444",
      wickUpColor: "#22C55E", wickDownColor: "#EF4444",
      borderVisible: false,
      // Same higher-precision price on the last-value label + price line as on the
      // axis; minMove lets the scale resolve to 1/100 of a cent.
      priceFormat: { type: "custom", formatter: fmtPrice, minMove: 0.0001 },
    });
    const volSeries = chart.addSeries(LC.HistogramSeries, {
      priceFormat: { type: "volume" }, priceScaleId: "", color: "#3F4659",
    });
    volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    // One markers primitive for the whole life of the chart; setOverlays() replaces its contents.
    const markersApi = LC.createSeriesMarkers(candleSeries, []);

    let overlaySeries = [];
    let bandSeries = [];
    let candleLen = 0;
    let fitPending = true;

    // ---- candles: fast path for a single trailing-bar change ----------------
    // update() only when the LAST candle changed in place or one bar was appended,
    // and never with a time older than what is already applied (lightweight-charts
    // throws "Cannot update oldest data"); otherwise a full setData().
    let lastCandleSig = "";
    let candleCount = 0;
    let appliedLastTime = null;

    function setCandles(candles) {
      candles = candles || [];
      candleLen = candles.length;
      if (candles.length === 0) {
        candleSeries.setData([]);
        appliedLastTime = null;
        candleCount = 0;
        lastCandleSig = "";
        // An empty dataset is the old "unmount": next data gets a fresh initial window.
        fitPending = true;
        resetRuler();
        return;
      }
      const last = candles[candles.length - 1];
      const sig = `${last.time}|${last.open}|${last.high}|${last.low}|${last.close}`;
      const lastTime = last.time;
      const prevLastTime = lastCandleSig ? Number(lastCandleSig.split("|")[0]) : null;
      const applied = appliedLastTime;

      const isInPlace = candleCount === candles.length && prevLastTime === lastTime;
      const isAppend = candleCount === candles.length - 1 && prevLastTime !== null && lastTime !== prevLastTime;
      const canFastUpdate = (isInPlace || isAppend) && (applied === null || lastTime >= applied);

      if (canFastUpdate) {
        candleSeries.update({ time: lastTime, open: last.open, high: last.high, low: last.low, close: last.close });
      } else {
        candleSeries.setData(candles.map((c) => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
      }
      appliedLastTime = lastTime;
      lastCandleSig = sig;
      candleCount = candles.length;
    }

    // ---- volume: same safe-update logic as candles --------------------------
    let lastVolSig = "";
    let volCount = 0;
    let appliedVolTime = null;

    function setVolume(volume) {
      volume = volume || [];
      if (volume.length === 0) {
        volSeries.setData([]);
        appliedVolTime = null;
        volCount = 0;
        lastVolSig = "";
        return;
      }
      const last = volume[volume.length - 1];
      const sig = `${last.time}|${last.value}`;
      if (volCount === volume.length && lastVolSig === sig) return; // tick didn't change the volume bar
      const applied = appliedVolTime;
      const isInPlace = volCount === volume.length;
      const isAppend = volCount === volume.length - 1;
      const canFastUpdate = (isInPlace || isAppend) && (applied === null || last.time >= applied);
      if (canFastUpdate) {
        volSeries.update({ time: last.time, value: last.value, color: last.color ?? "#2A3344" });
      } else {
        volSeries.setData(volume.map((v) => ({ time: v.time, value: v.value, color: v.color ?? "#2A3344" })));
      }
      appliedVolTime = last.time;
      lastVolSig = sig;
      volCount = volume.length;
    }

    // ---- overlays (SMA/EMA), bands (FVG / order-block edges), markers --------
    // Bands are two dashed line series (upper + lower edge): lightweight-charts OSS
    // has no native rectangle primitive.
    function setOverlays(overlays, bands, markers) {
      for (const s of overlaySeries) chart.removeSeries(s);
      for (const s of bandSeries) chart.removeSeries(s);
      overlaySeries = [];
      bandSeries = [];

      for (const o of overlays || []) {
        const s = chart.addSeries(LC.LineSeries, { color: o.color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
        s.setData(o.data.filter((p) => p.value != null).map((p) => ({ time: p.time, value: p.value })));
        overlaySeries.push(s);
      }
      for (const b of bands || []) {
        const opts = { color: b.color, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
        const upper = chart.addSeries(LC.LineSeries, opts);
        const lower = chart.addSeries(LC.LineSeries, opts);
        // A zone that starts on the last bar has start == end: one point (times must be ascending).
        const pts = (v) => (b.endTime > b.startTime
          ? [{ time: b.startTime, value: v }, { time: b.endTime, value: v }]
          : [{ time: b.startTime, value: v }]);
        upper.setData(pts(b.upper));
        lower.setData(pts(b.lower));
        bandSeries.push(upper, lower);
      }
      const m = (markers || [])
        .map((mk) => ({ time: mk.time, position: mk.position, color: mk.color, shape: mk.shape, text: mk.text }))
        .sort((a, b) => a.time - b.time);
      markersApi.setMarkers(m);
    }

    // ---- initial visible window: once per dataset, then leave zoom/pan alone --
    function fitIfPending(initialVisibleBars) {
      if (!fitPending || candleLen === 0) return;
      if (initialVisibleBars && candleLen > initialVisibleBars) {
        chart.timeScale().setVisibleLogicalRange({ from: candleLen - initialVisibleBars, to: candleLen });
      } else {
        chart.timeScale().fitContent();
      }
      fitPending = false;
    }

    function requestFit() {
      fitPending = true;
    }

    // ---- measurement ruler ----------------------------------------------------
    // Click 1 pins the start anchor, click 2 locks the measurement, click 3 resets.
    // While only one anchor is pinned the crosshair position drives a live preview.
    let rulerMode = false;
    let ruler = null;        // { start, end } locked
    let pendingStart = null; // first anchor only
    let hoverPoint = null;
    let subscribed = false;

    const hint = document.createElement("div");
    hint.className = "absolute top-2 left-2 z-10 rounded-md bg-surface2/90 backdrop-blur px-2 py-1 text-[11px] text-muted border border-line";
    hint.textContent = "ruler mode — click two points (3rd click resets)";
    hint.style.display = "none";
    el.appendChild(hint);

    // z-10: lightweight-charts' canvases sit at z-index 1-2, so without it the line
    // and anchor dots were painted underneath the chart (only the readout showed).
    const svg = svgEl("svg", { class: "absolute inset-0 z-10 pointer-events-none" });
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.display = "none";
    const line = svgEl("line", { "stroke-width": "1.5", "stroke-dasharray": "4 3" });
    const c1 = svgEl("circle", { r: "4" });
    const c2 = svgEl("circle", { r: "4" });
    svg.append(line, c1, c2);
    el.appendChild(svg);

    const label = document.createElement("div");
    label.style.display = "none";
    el.appendChild(label);

    function liveMeasurement() {
      if (ruler) return ruler;
      if (pendingStart && hoverPoint) return { start: pendingStart, end: { ...hoverPoint } };
      return null;
    }

    function renderRuler() {
      hint.style.display = rulerMode ? "" : "none";
      const m = liveMeasurement();
      if (!m) {
        svg.style.display = "none";
        label.style.display = "none";
        return;
      }
      const { start, end } = m;
      const dPrice = end.price - start.price;
      const dPct = start.price !== 0 ? (dPrice / start.price) * 100 : 0;
      const dSecs = end.time - start.time;
      const bars = Math.max(1, Math.round(dSecs / 86400)); // days, approximate for daily charts
      const positive = dPct >= 0;
      const color = positive ? "#22C55E" : "#EF4444";

      line.setAttribute("x1", start.x);
      line.setAttribute("y1", start.y);
      line.setAttribute("x2", end.x);
      line.setAttribute("y2", end.y);
      line.setAttribute("stroke", color);
      c1.setAttribute("cx", start.x);
      c1.setAttribute("cy", start.y);
      c1.setAttribute("fill", color);
      c2.setAttribute("cx", end.x);
      c2.setAttribute("cy", end.y);
      c2.setAttribute("fill", color);
      svg.style.display = "";

      // Readout midway between the anchors, just above the higher one.
      label.className = `absolute z-20 -translate-x-1/2 -translate-y-full rounded-md px-2 py-1 text-xs font-mono tabular border whitespace-nowrap ${
        positive
          ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-300"
          : "bg-rose-500/15 border-rose-500/40 text-rose-300"
      }`;
      label.style.left = `${(start.x + end.x) / 2}px`;
      label.style.top = `${Math.min(start.y, end.y) - 8}px`;
      const sign = positive ? "+" : "";
      label.textContent = `${sign}${dPct.toFixed(2)}% · ${sign}${dPrice.toFixed(2)} · ${bars} bar${bars === 1 ? "" : "s"}`;
      label.style.display = "";
    }

    function resetRuler() {
      ruler = null;
      pendingStart = null;
      hoverPoint = null;
      renderRuler();
    }

    function onClick(param) {
      if (!param.point || !param.time) return;
      const price = candleSeries.coordinateToPrice(param.point.y);
      if (price == null) return;
      const a = { time: Number(param.time), price: Number(price), x: param.point.x, y: param.point.y };
      if (ruler) {
        // third click: reset
        ruler = null;
        pendingStart = null;
      } else if (!pendingStart) {
        pendingStart = a;
      } else {
        ruler = { start: pendingStart, end: a };
        pendingStart = null;
      }
      renderRuler();
    }

    function onMove(param) {
      if (!param.point || !param.time) {
        hoverPoint = null;
      } else {
        const price = candleSeries.coordinateToPrice(param.point.y);
        hoverPoint = price == null ? null : { x: param.point.x, y: param.point.y, price: Number(price), time: Number(param.time) };
      }
      renderRuler();
    }

    // A FREE (Normal) crosshair while the ruler is on, so it can sit at any price;
    // Magnet (snaps to OHLC, nice for reading exact values) when it is off.
    function setRulerMode(on) {
      rulerMode = !!on;
      chart.applyOptions({ crosshair: { mode: rulerMode ? LC.CrosshairMode.Normal : LC.CrosshairMode.Magnet } });
      if (rulerMode && !subscribed) {
        chart.subscribeClick(onClick);
        chart.subscribeCrosshairMove(onMove);
        subscribed = true;
      } else if (!rulerMode && subscribed) {
        chart.unsubscribeClick(onClick);
        chart.unsubscribeCrosshairMove(onMove);
        subscribed = false;
      }
      resetRuler();
    }

    function destroy() {
      ro.disconnect();
      if (subscribed) {
        chart.unsubscribeClick(onClick);
        chart.unsubscribeCrosshairMove(onMove);
      }
      chart.remove();
      el.innerHTML = "";
      delete el._candleChart;
    }

    const api = { setCandles, setVolume, setOverlays, fitIfPending, requestFit, setRulerMode, destroy };
    el._candleChart = api;
    return api;
  }

  window.createCandlestickChart = createCandlestickChart;
})();
