"""Sidebar navigation and the page registry.

One source of truth: every entry here becomes a sidebar link, and every page in
PAGES becomes a route served by `main.py`. Icons are Lucide names.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    href: str
    icon: str
    label: str
    hint: str
    match: str | None = None  # path prefix that also marks the item active

    def is_active(self, path: str) -> bool:
        return path == self.href or path.startswith(self.match or f"{self.href}/")


@dataclass(frozen=True)
class NavGroup:
    title: str | None
    items: tuple[NavItem, ...]


# Grouped by what you're trying to do. `hint` shows on hover so each page explains itself.
NAV_GROUPS: tuple[NavGroup, ...] = (
    NavGroup(None, (
        NavItem("/dashboard", "home", "Dashboard", "Morning check: market mood, indices, your watchlist, news"),
        NavItem("/stocks/AAPL", "line-chart", "Stock analysis",
                "Chart, AI opinion and smart-money read for one symbol (search at the top)", match="/stocks/"),
        NavItem("/watchlists", "eye", "Watchlists", "Lists of symbols you follow"),
        NavItem("/portfolio", "briefcase", "Portfolio", "Your holdings, P&L and allocation"),
    )),
    NavGroup("Find ideas", (
        NavItem("/opportunities", "target", "Opportunities",
                "Live feed from 6 strategies, ranked by how their past picks did vs SPY"),
        NavItem("/rising-stars", "trending-up", "Rising stars",
                "One of the Opportunities strategies on its own, with a market-cap filter"),
        NavItem("/screener", "filter", "Screener", "Browse the symbol universe by asset class and sector"),
        NavItem("/compare", "git-compare", "Compare", "Several symbols side by side"),
        NavItem("/news", "newspaper", "News", "Market news with a sentiment score"),
        NavItem("/congress", "landmark", "Congress trades",
                "Stock trades disclosed by US House members, and what copying them earned"),
    )),
    NavGroup("Test & automate", (
        NavItem("/strategy-lab", "flask-conical", "Strategy Lab", "Backtest a simple indicator strategy on one symbol"),
        NavItem("/swing", "waves", "Swing trading", "Multi-day setups: today's scan and portfolio backtests"),
        NavItem("/trading-bot", "bot", "Trading bots", "Your bots, their paper trades and the AI risk-gate"),
        NavItem("/live-test", "activity", "Live test (paper)",
                "Monthly momentum strategy running forward with fake money vs SPY"),
    )),
    NavGroup("Is it working?", (
        NavItem("/track-record", "trophy", "Picks track record", "How the Opportunities picks did 1, 4 and 12 weeks later"),
        NavItem("/calibration", "gauge", "Forecast accuracy", "How accurate the AI price forecasts turned out"),
    )),
)

FOOTER_ITEMS: tuple[NavItem, ...] = (
    NavItem("/ai-chat", "message-square", "AI Assistant", "Ask questions; it answers using the platform's own data"),
    NavItem("/settings", "settings", "Settings", "Account and 2FA"),
)

# path -> (template, browser tab title). `/stocks/{symbol}` is routed separately.
PAGES: dict[str, tuple[str, str]] = {
    "/dashboard": ("pages/dashboard.html", "Dashboard"),
    "/watchlists": ("pages/watchlists.html", "Watchlists"),
    "/portfolio": ("pages/portfolio.html", "Portfolio"),
    "/opportunities": ("pages/opportunities.html", "Opportunities"),
    "/rising-stars": ("pages/rising_stars.html", "Rising stars"),
    "/screener": ("pages/screener.html", "Screener"),
    "/compare": ("pages/compare.html", "Compare"),
    "/news": ("pages/news.html", "News"),
    "/congress": ("pages/congress.html", "Congress trades"),
    "/strategy-lab": ("pages/strategy_lab.html", "Strategy Lab"),
    "/swing": ("pages/swing.html", "Swing trading"),
    "/trading-bot": ("pages/trading_bot.html", "Trading bots"),
    "/live-test": ("pages/live_test.html", "Live test"),
    "/track-record": ("pages/track_record.html", "Track record"),
    "/calibration": ("pages/calibration.html", "Forecast accuracy"),
    "/ai-chat": ("pages/ai_chat.html", "AI Assistant"),
    "/settings": ("pages/settings.html", "Settings"),
    "/login": ("pages/login.html", "Sign in"),
    "/register": ("pages/register.html", "Sign up"),
}
