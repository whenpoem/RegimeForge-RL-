from __future__ import annotations

import argparse
import locale
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None

from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import REGIME_LABELS
from .training import TrainingManager


ASCII_SPARK = ".:-=+*#%@"
UNICODE_SPARK = "▁▂▃▄▅▆▇█"
ASCII_HEAT = " .:-#"
UNICODE_HEAT = " ░▒▓█"
THEME = {
    "accent": "#6366f1",
    "accent_alt": "#8b5cf6",
    "good": "#10b981",
    "bad": "#ef4444",
    "warn": "#f59e0b",
    "shock": "#ec4899",
    "cyan": "#22d3ee",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "line": "#334155",
}
REGIME_COLORS = {
    "bull": THEME["good"],
    "bear": THEME["bad"],
    "chop": THEME["warn"],
    "shock": THEME["shock"],
}
EXPERT_COLORS = ["#22d3ee", "#fb923c", "#c084fc", "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#f87171"]
SPARK_COLORS = ["#ef4444", "#f97316", "#facc15", "#84cc16", "#10b981"]
LOSS_COLORS = ["#38bdf8", "#60a5fa", "#818cf8", "#a78bfa", "#c084fc"]
VIEW_IDS = ("overview", "regime", "expert", "performance", "config")
FOCUS = {
    "overview": ["hero", "recent", "regime", "performance", "checkpoint", "expert"],
    "regime": ["hero", "regime", "analysis", "timeline"],
    "expert": ["hero", "expert", "heatmap", "specialisation"],
    "performance": ["hero", "performance", "baseline", "regimes"],
    "config": ["hero", "config", "runtime", "controls"],
}
I18N = {
    "en": {
        "overview": "Overview",
        "regime": "Regime Lens",
        "expert": "Expert Deep Dive",
        "performance": "Performance",
        "config": "Config",
        "recent": "Recent Training",
        "checkpoint": "Checkpoint Baselines",
        "runtime": "Runtime + Config",
        "performance_panel": "Performance Dashboard",
        "expert_panel": "Expert Activation",
        "controls": "Controls",
        "no_run": "No run available yet.",
        "waiting": "Waiting for training to start...",
        "no_checkpoint": "No checkpoint yet.",
        "not_rcmoe": "This view becomes fully active with RCMoE-DQN.",
        "reward": "Reward",
        "return": "Return",
        "loss": "Loss",
        "epsilon": "Epsilon",
        "elapsed": "Elapsed",
        "episodes": "episodes",
        "steps": "steps",
        "gate_acc": "Gate Acc",
        "ground_truth": "Ground Truth",
        "rolling_acc": "Rolling Accuracy",
        "keys": "Keys",
        "focus": "Focus",
        "view": "View",
        "paused": "PAUSED",
        "running": "RUNNING",
        "done": "DONE",
        "failed": "FAILED",
        "press_exit": "Press q, Esc, or Ctrl+C to exit.",
    },
    "zh": {
        "overview": "总览",
        "regime": "Regime Lens",
        "expert": "专家深潜",
        "performance": "绩效",
        "config": "配置",
        "recent": "近期训练",
        "checkpoint": "Checkpoint 基线对比",
        "runtime": "运行时与配置",
        "performance_panel": "绩效仪表盘",
        "expert_panel": "专家激活",
        "controls": "交互说明",
        "no_run": "当前还没有可用运行。",
        "waiting": "等待训练启动中……",
        "no_checkpoint": "还没有 checkpoint。",
        "not_rcmoe": "该视图在 agent 使用 RCMoE-DQN 时会完全激活。",
        "reward": "奖励",
        "return": "收益",
        "loss": "损失",
        "epsilon": "探索率",
        "elapsed": "已用时",
        "episodes": "轮",
        "steps": "步",
        "gate_acc": "Gate 准确率",
        "ground_truth": "真实 regime",
        "rolling_acc": "滚动准确率",
        "keys": "快捷键",
        "focus": "焦点",
        "view": "视图",
        "paused": "已暂停",
        "running": "运行中",
        "done": "已完成",
        "failed": "失败",
        "press_exit": "按 q、Esc 或 Ctrl+C 退出。",
    },
}
_GPU_CACHE = {"ts": 0.0, "value": "n/a"}


@dataclass(slots=True)
class DashboardState:
    view_index: int = 0
    focus_index: int = 0
    show_regime_detail: bool = True
    show_expert_detail: bool = True
    should_exit: bool = False

    @property
    def view_id(self) -> str:
        return VIEW_IDS[self.view_index]


class KeyboardReader:
    def __init__(self) -> None:
        self.fd: int | None = None
        self.settings: Any = None

    def __enter__(self) -> "KeyboardReader":
        if msvcrt is None and sys.stdin.isatty():
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None and self.settings is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)

    def read_key(self) -> str | None:
        if msvcrt is not None:
            if not msvcrt.kbhit():
                return None
            first = msvcrt.getwch()
            if first in ("\x00", "\xe0"):
                return "shift_tab" if msvcrt.getwch() == "\x0f" else None
            if first == "\t":
                return "tab"
            if first == " ":
                return "space"
            if first == "\x1b":
                return "escape"
            return first.lower()
        if not sys.stdin.isatty():
            return None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        first = sys.stdin.read(1)
        if first == "\t":
            return "tab"
        if first == " ":
            return "space"
        if first == "\x1b":
            return "escape"
        return first.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime Lens terminal dashboard")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh training run instead of watching the latest run.")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Dashboard refresh interval.")
    parser.add_argument("--fullscreen", action="store_true", help="Use the terminal alternate screen mode.")
    parser.add_argument("--lang", choices=("auto", "en", "zh"), default="auto")
    parser.add_argument("--charset", choices=("auto", "unicode", "ascii"), default="auto")
    args = parser.parse_args()

    lang = _lang(args.lang)
    charset = _charset(args.charset)
    manager = TrainingManager()
    state = DashboardState()
    if args.fresh or manager.current_run_id is None:
        manager.start_new_run()
    else:
        manager.maybe_autostart()

    console = Console()
    refresh = max(2, math.ceil(1 / max(args.poll_seconds, 0.15)))
    with KeyboardReader() as reader, Live(_build_dashboard(manager, state, lang, charset), refresh_per_second=refresh, screen=args.fullscreen, console=console) as live:
        try:
            while not state.should_exit:
                key = reader.read_key()
                if key:
                    _handle_key(key, state, manager)
                live.update(_build_dashboard(manager, state, lang, charset))
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            pass


def _handle_key(key: str, state: DashboardState, manager: TrainingManager) -> None:
    if key in {"q", "escape"}:
        state.should_exit = True
    elif key in {"1", "2", "3", "4", "5"}:
        state.view_index = int(key) - 1
        state.focus_index = 0
    elif key == "tab":
        state.focus_index = (state.focus_index + 1) % len(FOCUS[state.view_id])
    elif key == "shift_tab":
        state.focus_index = (state.focus_index - 1) % len(FOCUS[state.view_id])
    elif key == "space":
        manager.toggle_pause()
    elif key == "r":
        state.show_regime_detail = not state.show_regime_detail
    elif key == "e":
        state.show_expert_detail = not state.show_expert_detail


def _build_dashboard(manager: TrainingManager, state: DashboardState, lang: str, charset: str) -> Layout:
    run = manager.latest_run_summary()
    if run is None:
        layout = Layout()
        layout.split_column(Layout(_empty_panel(_t(lang, "no_run"), "Regime Lens", THEME["accent"])), Layout(_status_bar(None, manager, state, lang), size=3))
        return layout

    metrics = manager.metrics() or {"series": []}
    checkpoints = manager.checkpoints() or {"checkpoints": []}
    series = metrics.get("series", [])
    snapshot = manager.live_snapshot()
    latest_metric = series[-1] if series else None
    latest_checkpoint = checkpoints["checkpoints"][-1] if checkpoints["checkpoints"] else None
    ckpt_id = str(latest_checkpoint["checkpointId"]) if latest_checkpoint else None
    regime_analysis = manager.checkpoint_regime_analysis(ckpt_id) if ckpt_id else None
    expert_analysis = manager.checkpoint_expert_analysis(ckpt_id) if ckpt_id else None

    if state.view_id == "overview":
        layout = _overview_layout(run, series, latest_metric, latest_checkpoint, snapshot, regime_analysis, expert_analysis, state, lang, charset)
    elif state.view_id == "regime":
        layout = _regime_layout(run, latest_metric, latest_checkpoint, snapshot, regime_analysis, state, lang, charset)
    elif state.view_id == "expert":
        layout = _expert_layout(run, latest_metric, snapshot, regime_analysis, expert_analysis, state, lang, charset)
    elif state.view_id == "performance":
        layout = _performance_layout(run, latest_metric, latest_checkpoint, snapshot, state, lang, charset)
    else:
        layout = _config_layout(run, latest_metric, manager, state, lang)

    wrapper = Layout()
    wrapper.split_column(layout, Layout(_status_bar(run, manager, state, lang), size=3))
    return wrapper


def _overview_layout(run: dict[str, Any], series: list[dict[str, Any]], latest_metric: dict[str, Any] | None, latest_checkpoint: dict[str, Any] | None, snapshot: dict[str, Any], regime_analysis: dict[str, Any] | None, expert_analysis: dict[str, Any] | None, state: DashboardState, lang: str, charset: str) -> Layout:
    layout = Layout()
    layout.split_column(Layout(_hero(run, latest_metric, latest_checkpoint, snapshot, state, "hero", lang, charset), size=9), Layout(name="body"))
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))
    layout["body"]["left"].split_column(
        Layout(_recent_panel(series, latest_metric, state, "recent", lang, charset), ratio=1),
        Layout(_regime_panel(snapshot, regime_analysis, state, "regime", lang, charset), ratio=1),
    )
    layout["body"]["right"].split_column(
        Layout(_performance_panel(latest_checkpoint, snapshot, state, "performance", lang, charset), ratio=1),
        Layout(name="bottom", ratio=1),
    )
    layout["body"]["right"]["bottom"].split_row(
        Layout(_checkpoint_panel(latest_checkpoint, state, "checkpoint", lang, charset), ratio=1),
        Layout(_expert_panel(snapshot, expert_analysis, state, "expert", lang, charset), ratio=1),
    )
    return layout


def _regime_layout(run: dict[str, Any], latest_metric: dict[str, Any] | None, latest_checkpoint: dict[str, Any] | None, snapshot: dict[str, Any], regime_analysis: dict[str, Any] | None, state: DashboardState, lang: str, charset: str) -> Layout:
    layout = Layout()
    layout.split_column(Layout(_hero(run, latest_metric, latest_checkpoint, snapshot, state, "hero", lang, charset), size=9), Layout(name="body"))
    layout["body"].split_row(
        Layout(_regime_panel(snapshot, regime_analysis, state, "regime", lang, charset), ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["body"]["right"].split_column(
        Layout(_analysis_panel(regime_analysis, snapshot, state, "analysis", lang), ratio=1),
        Layout(_timeline_panel(snapshot, state, "timeline", lang), ratio=1),
    )
    return layout


def _expert_layout(run: dict[str, Any], latest_metric: dict[str, Any] | None, snapshot: dict[str, Any], regime_analysis: dict[str, Any] | None, expert_analysis: dict[str, Any] | None, state: DashboardState, lang: str, charset: str) -> Layout:
    layout = Layout()
    layout.split_column(Layout(_hero(run, latest_metric, None, snapshot, state, "hero", lang, charset), size=9), Layout(name="body"))
    layout["body"].split_row(
        Layout(_expert_panel(snapshot, expert_analysis, state, "expert", lang, charset), ratio=2),
        Layout(name="right", ratio=3),
    )
    layout["body"]["right"].split_column(
        Layout(_heatmap_panel(expert_analysis, state, "heatmap", charset), ratio=1),
        Layout(_specialisation_panel(expert_analysis, regime_analysis, state, "specialisation", charset), ratio=1),
    )
    return layout


def _performance_layout(run: dict[str, Any], latest_metric: dict[str, Any] | None, latest_checkpoint: dict[str, Any] | None, snapshot: dict[str, Any], state: DashboardState, lang: str, charset: str) -> Layout:
    layout = Layout()
    layout.split_column(Layout(_hero(run, latest_metric, latest_checkpoint, snapshot, state, "hero", lang, charset), size=9), Layout(name="body"))
    layout["body"].split_row(
        Layout(_performance_panel(latest_checkpoint, snapshot, state, "performance", lang, charset), ratio=2),
        Layout(name="right", ratio=3),
    )
    layout["body"]["right"].split_column(
        Layout(_checkpoint_panel(latest_checkpoint, state, "baseline", lang, charset), ratio=1),
        Layout(_per_regime_panel(latest_checkpoint, state, "regimes", charset), ratio=1),
    )
    return layout


def _config_layout(run: dict[str, Any], latest_metric: dict[str, Any] | None, manager: TrainingManager, state: DashboardState, lang: str) -> Layout:
    layout = Layout()
    layout.split_column(Layout(_hero(run, latest_metric, None, manager.live_snapshot(), state, "hero", lang, "unicode"), size=9), Layout(name="body"))
    layout["body"].split_row(Layout(_config_panel(run, state, "config"), ratio=3), Layout(name="right", ratio=2))
    layout["body"]["right"].split_column(
        Layout(_runtime_panel(run, latest_metric, state, "runtime", lang), ratio=1),
        Layout(_controls_panel(manager, state, "controls", lang), ratio=1),
    )
    return layout


def _hero(run: dict[str, Any], latest_metric: dict[str, Any] | None, latest_checkpoint: dict[str, Any] | None, snapshot: dict[str, Any], state: DashboardState, panel_id: str, lang: str, charset: str) -> Panel:
    cfg = run.get("config", {})
    runtime = run.get("runtime", {})
    current = int(run.get("currentEpisode", 0))
    total = int(run.get("episodesPlanned") or cfg.get("episodes", 0) or 0)
    steps = int(run.get("globalStep", 0))
    elapsed = float(run.get("elapsedSeconds", 0.0) or 0.0)
    rate = steps / elapsed if elapsed > 0 else 0.0

    title = Text("Regime Lens", style=f"bold {THEME['text']}")
    title.append("  ")
    title.append(_view_title(state.view_id, lang, charset), style=f"bold {THEME['accent_alt']}")
    badges = Text()
    badges.append(_badge(_status_text(run, snapshot, lang), _status_color(run, snapshot)))
    badges.append(" ")
    badges.append(_badge(str(run.get("agentType", cfg.get("agent_type", "dqn"))).upper(), THEME["accent"]))
    badges.append(" ")
    badges.append(_badge(str(runtime.get("resolved_device", cfg.get("device", "auto"))).upper(), THEME["cyan"]))

    summary = Text()
    summary.append(f"{current}/{total} {_t(lang, 'episodes')}", style=f"bold {THEME['text']}")
    summary.append("  |  ", style=THEME["muted"])
    summary.append(f"{steps:,} {_t(lang, 'steps')}", style=f"bold {THEME['text']}")
    summary.append("  |  ", style=THEME["muted"])
    summary.append(f"{rate:,.0f} step/s", style=THEME["good"])
    if latest_checkpoint:
        summary.append("  |  ", style=THEME["muted"])
        summary.append(str(latest_checkpoint.get("checkpointId", "--")), style=THEME["accent_alt"])

    gate_acc = snapshot.get("gate_accuracy")
    if snapshot.get("gate_weights") is None:
        gate_acc = _get(latest_metric, "gateAccuracy", gate_acc)
    cards = Columns(
        [
            _stat("Reward", _pct(_get(latest_metric, "totalReward")), _value_color(_get(latest_metric, "totalReward"))),
            _stat("Return", _pct(_get(latest_metric, "strategyReturn")), _value_color(_get(latest_metric, "strategyReturn"))),
            _stat(_t(lang, "epsilon"), _num(_get(latest_metric, "epsilon"), 3), THEME["warn"]),
            _stat(_t(lang, "elapsed"), _duration(elapsed), THEME["text"]),
            _stat(_t(lang, "gate_acc"), _pct(gate_acc), THEME["cyan"]),
        ],
        expand=True,
        equal=True,
    )
    body = Group(title, badges, summary, _bar(current / max(total, 1), THEME["accent"], charset, 44), cards)
    return _panel(_t(lang, "overview"), body, state, panel_id, THEME["accent"])


def _recent_panel(series: list[dict[str, Any]], latest_metric: dict[str, Any] | None, state: DashboardState, panel_id: str, lang: str, charset: str) -> Panel:
    rows = Table.grid(expand=True)
    rows.add_column(width=9)
    rows.add_column(ratio=3)
    rows.add_column(width=15)
    window = series[-80:] if series else ([] if latest_metric is None else [latest_metric])
    reward = [(_get(item, "totalReward", 0.0) or 0.0) * 100.0 for item in window]
    ret = [(_get(item, "strategyReturn", 0.0) or 0.0) * 100.0 for item in window]
    loss = [(_get(item, "avgLoss", 0.0) or 0.0) for item in window if item.get("avgLoss") is not None]
    eps = [(_get(item, "epsilon", 0.0) or 0.0) for item in window]
    rows.add_row(_t(lang, "reward"), _spark(reward, SPARK_COLORS, charset), Text(_pct(_get(latest_metric, "totalReward")), style=_value_color(_get(latest_metric, "totalReward"))))
    rows.add_row(_t(lang, "return"), _spark(ret, SPARK_COLORS, charset), Text(_pct(_get(latest_metric, "strategyReturn")), style=_value_color(_get(latest_metric, "strategyReturn"))))
    rows.add_row(_t(lang, "loss"), _spark(loss, LOSS_COLORS, charset), Text(_num(_get(latest_metric, "avgLoss"), 4), style=THEME["accent_alt"]))
    rows.add_row(_t(lang, "epsilon"), _spark(eps, LOSS_COLORS, charset), Text(_num(_get(latest_metric, "epsilon"), 3), style=THEME["warn"]))
    footer = Text(_t(lang, "press_exit"), style=THEME["muted"])
    return _panel(_t(lang, "recent"), Group(rows, footer), state, panel_id, THEME["good"])


def _checkpoint_panel(latest_checkpoint: dict[str, Any] | None, state: DashboardState, panel_id: str, lang: str, charset: str) -> Panel:
    if latest_checkpoint is None:
        return _panel(_t(lang, "checkpoint"), Text(_t(lang, "no_checkpoint"), style=THEME["muted"]), state, panel_id, THEME["accent_alt"])
    agent_return = float(latest_checkpoint.get("agentReturn", 0.0))
    random_return = float(latest_checkpoint.get("randomReturn", 0.0))
    hold_return = float(latest_checkpoint.get("buyHoldReturn", 0.0))
    reward = float(latest_checkpoint.get("avgEvalReward", 0.0))
    rows = Table.grid(expand=True)
    rows.add_column(width=13)
    rows.add_column(width=11, justify="right")
    rows.add_column(ratio=2)
    rows.add_row("Agent", _pct(agent_return), _bar(min(abs(agent_return) / 0.12, 1.0), THEME["good"] if agent_return >= 0 else THEME["bad"], charset, 10))
    rows.add_row("Random", _pct(random_return), _bar(min(abs(random_return) / 0.12, 1.0), THEME["warn"] if random_return >= 0 else THEME["bad"], charset, 10))
    rows.add_row("Buy&Hold", _pct(hold_return), _bar(min(abs(hold_return) / 0.12, 1.0), THEME["cyan"] if hold_return >= 0 else THEME["bad"], charset, 10))
    rows.add_row("Eval Reward", _pct(reward), _bar(min(abs(reward) / 0.12, 1.0), THEME["accent"], charset, 10))
    delta = Text()
    delta.append(f"Agent-Random {_pct(agent_return - random_return)}", style=_value_color(agent_return - random_return))
    delta.append("  |  ", style=THEME["muted"])
    delta.append(f"Agent-Hold {_pct(agent_return - hold_return)}", style=_value_color(agent_return - hold_return))
    return _panel(_t(lang, "checkpoint"), Group(rows, delta), state, panel_id, THEME["accent_alt"])


def _regime_panel(snapshot: dict[str, Any], regime_analysis: dict[str, Any] | None, state: DashboardState, panel_id: str, lang: str, charset: str) -> Panel:
    weights = snapshot.get("gate_weights")
    if weights is None and regime_analysis is None:
        return _panel(_t(lang, "regime"), Text(_t(lang, "not_rcmoe"), style=THEME["muted"]), state, panel_id, THEME["shock"])
    if weights is None and regime_analysis is not None:
        info = Table.grid(expand=True)
        info.add_column()
        info.add_column(justify="right")
        info.add_row("NMI", _num(regime_analysis.get("nmi"), 3))
        info.add_row("ARI", _num(regime_analysis.get("ari"), 3))
        info.add_row("Entropy", _num(regime_analysis.get("gate_entropy"), 3))
        info.add_row("Probe", _num(regime_analysis.get("probing_accuracy"), 3))
        note = Text("checkpoint regime analysis fallback", style=THEME["muted"])
        return _panel(_t(lang, "regime"), Group(info, note), state, panel_id, THEME["shock"])
    table = Table.grid(expand=True)
    table.add_column(width=10)
    table.add_column(ratio=3)
    table.add_column(width=8, justify="right")
    n = min(len(weights), len(REGIME_LABELS))
    for index in range(n):
        label = REGIME_LABELS[index]
        table.add_row(label.title(), _bar(float(weights[index]), REGIME_COLORS[label], charset, 18), Text(f"{float(weights[index]) * 100:5.1f}%", style=REGIME_COLORS[label]))
    for index in range(n, len(weights)):
        color = EXPERT_COLORS[index % len(EXPERT_COLORS)]
        table.add_row(f"E{index + 1}", _bar(float(weights[index]), color, charset, 18), Text(f"{float(weights[index]) * 100:5.1f}%", style=color))
    dominant = int(weights.argmax())
    regime_index = int(snapshot.get("regime_index", -1))
    truth = Text()
    truth.append(f"{_t(lang, 'ground_truth')}: ", style=f"bold {THEME['text']}")
    truth.append(str(snapshot.get("regime", "--")).upper(), style=REGIME_COLORS.get(str(snapshot.get("regime", "")), THEME["muted"]))
    truth.append("  |  ", style=THEME["muted"])
    truth.append(f"{_t(lang, 'rolling_acc')}: {_pct(snapshot.get('gate_accuracy'))}", style=THEME["cyan"])
    truth.append("  |  ", style=THEME["muted"])
    truth.append("✓" if dominant == regime_index else "✗", style=THEME["good"] if dominant == regime_index else THEME["bad"])
    items: list[Any] = [table, truth]
    if state.show_regime_detail and regime_analysis:
        info = Text(f"NMI {float(regime_analysis.get('nmi', 0.0)):.3f}  |  ARI {float(regime_analysis.get('ari', 0.0)):.3f}  |  Entropy {float(regime_analysis.get('gate_entropy', 0.0)):.3f}", style=THEME["accent"])
        items.append(info)
    return _panel(_t(lang, "regime"), Group(*items), state, panel_id, THEME["shock"])


def _analysis_panel(regime_analysis: dict[str, Any] | None, snapshot: dict[str, Any], state: DashboardState, panel_id: str, lang: str) -> Panel:
    if regime_analysis is None:
        return _panel("Analysis", Text(_t(lang, "no_checkpoint"), style=THEME["muted"]), state, panel_id, THEME["accent"])
    rows = Table.grid(expand=True)
    rows.add_column()
    rows.add_column(justify="right")
    rows.add_row("NMI", _num(regime_analysis.get("nmi"), 4))
    rows.add_row("ARI", _num(regime_analysis.get("ari"), 4))
    rows.add_row("Entropy", _num(regime_analysis.get("gate_entropy"), 4))
    rows.add_row("Probe", _num(regime_analysis.get("probing_accuracy"), 4))
    rows.add_row("Separation", _num(regime_analysis.get("separation_score"), 4))
    rows.add_row(_t(lang, "gate_acc"), _pct(snapshot.get("gate_accuracy")))
    return _panel("Analysis", rows, state, panel_id, THEME["accent"])


def _timeline_panel(snapshot: dict[str, Any], state: DashboardState, panel_id: str, lang: str) -> Panel:
    history = snapshot.get("expert_history", [])
    if not history:
        return _panel("Timeline", Text(_t(lang, "waiting"), style=THEME["muted"]), state, panel_id, THEME["warn"])
    recent = history[-40:]
    text = Text()
    for item in recent:
        weights = item.get("weights", [])
        if not weights:
            continue
        pred = int(max(range(len(weights)), key=lambda idx: weights[idx]))
        ok = pred == int(item.get("regime_index", -1))
        text.append(str(pred % 10), style=THEME["good"] if ok else THEME["bad"])
    note = Text("recent route trace", style=THEME["muted"])
    return _panel("Timeline", Group(text, note), state, panel_id, THEME["warn"])


def _expert_panel(snapshot: dict[str, Any], expert_analysis: dict[str, Any] | None, state: DashboardState, panel_id: str, lang: str, charset: str) -> Panel:
    weights = snapshot.get("gate_weights")
    if weights is None and expert_analysis is None:
        return _panel(_t(lang, "expert_panel"), Text(_t(lang, "not_rcmoe"), style=THEME["muted"]), state, panel_id, THEME["cyan"])
    if weights is None and expert_analysis is not None:
        util = expert_analysis.get("expert_utilization", [])
        table = Table.grid(expand=True)
        table.add_column(width=9)
        table.add_column(ratio=3)
        table.add_column(width=8, justify="right")
        for index, value in enumerate(util):
            color = EXPERT_COLORS[index % len(EXPERT_COLORS)]
            table.add_row(f"E{index + 1}", _bar(float(value), color, charset, 18), Text(f"{float(value) * 100:5.1f}%", style=color))
        footer = Text(f"Score {_num(expert_analysis.get('specialisation_score'), 3)}", style=THEME['accent'])
        return _panel(_t(lang, "expert_panel"), Group(table, footer), state, panel_id, THEME["cyan"])
    table = Table.grid(expand=True)
    table.add_column(width=9)
    table.add_column(ratio=3)
    table.add_column(width=8, justify="right")
    for index, weight in enumerate(weights):
        color = EXPERT_COLORS[index % len(EXPERT_COLORS)]
        table.add_row(f"E{index + 1}", _bar(float(weight), color, charset, 18), Text(f"{float(weight) * 100:5.1f}%", style=color))
    footer = Text(f"Dominant E{int(weights.argmax()) + 1}", style=THEME["accent"])
    if expert_analysis and state.show_expert_detail:
        util = expert_analysis.get("expert_utilization", [])
        footer.append("  |  ", style=THEME["muted"])
        footer.append(", ".join(f"E{i + 1}:{float(value) * 100:.1f}%" for i, value in enumerate(util)), style=THEME["cyan"])
    return _panel(_t(lang, "expert_panel"), Group(table, footer), state, panel_id, THEME["cyan"])


def _heatmap_panel(expert_analysis: dict[str, Any] | None, state: DashboardState, panel_id: str, charset: str) -> Panel:
    if expert_analysis is None:
        return _panel("Heatmap", Text("No checkpoint", style=THEME["muted"]), state, panel_id, THEME["accent_alt"])
    matrix = expert_analysis.get("activation_matrix", [])
    if not matrix:
        return _panel("Heatmap", Text("No data", style=THEME["muted"]), state, panel_id, THEME["accent_alt"])
    chars = UNICODE_HEAT if charset == "unicode" else ASCII_HEAT
    table = Table.grid(expand=True)
    table.add_column(width=8)
    for _ in range(len(matrix[0])):
        table.add_column(width=4, justify="center")
    table.add_row("", *[f"E{i + 1}" for i in range(len(matrix[0]))])
    for label, row in zip(REGIME_LABELS, matrix, strict=False):
        cells: list[Any] = [Text(label.title(), style=REGIME_COLORS[label])]
        for index, value in enumerate(row):
            level = min(len(chars) - 1, max(0, int(round(float(value) * (len(chars) - 1)))))
            cells.append(Text(chars[level] * 2, style=EXPERT_COLORS[index % len(EXPERT_COLORS)]))
        table.add_row(*cells)
    return _panel("Heatmap", table, state, panel_id, THEME["accent_alt"])


def _specialisation_panel(expert_analysis: dict[str, Any] | None, regime_analysis: dict[str, Any] | None, state: DashboardState, panel_id: str, charset: str) -> Panel:
    if expert_analysis is None:
        return _panel("Specialisation", Text("No checkpoint", style=THEME["muted"]), state, panel_id, THEME["warn"])
    util = expert_analysis.get("expert_utilization", [])
    rows = Table.grid(expand=True)
    rows.add_column(width=8)
    rows.add_column(width=8, justify="right")
    rows.add_column(ratio=2)
    for index, value in enumerate(util):
        color = EXPERT_COLORS[index % len(EXPERT_COLORS)]
        rows.add_row(f"E{index + 1}", f"{float(value) * 100:5.1f}%", _bar(float(value), color, charset, 10))
    footer = Text(f"Score {float(expert_analysis.get('specialisation_score', 0.0)):.3f}", style=THEME["accent"])
    if regime_analysis:
        footer.append("  |  ", style=THEME["muted"])
        footer.append(f"NMI {float(regime_analysis.get('nmi', 0.0)):.3f}", style=THEME["accent_alt"])
    return _panel("Specialisation", Group(rows, footer), state, panel_id, THEME["warn"])


def _performance_panel(latest_checkpoint: dict[str, Any] | None, snapshot: dict[str, Any], state: DashboardState, panel_id: str, lang: str, charset: str) -> Panel:
    live = snapshot.get("financial_metrics", {})
    ckpt = latest_checkpoint.get("financialMetrics", {}) if latest_checkpoint else {}
    sharpe = live.get("sharpe", ckpt.get("sharpe") if isinstance(ckpt, dict) else None)
    sortino = live.get("sortino", ckpt.get("sortino") if isinstance(ckpt, dict) else None)
    max_drawdown = live.get("max_drawdown", ckpt.get("max_drawdown") if isinstance(ckpt, dict) else None)
    win_rate = live.get("win_rate", ckpt.get("win_rate") if isinstance(ckpt, dict) else None)
    cards = Columns(
        [
            _stat("Sharpe", _num(sharpe, 3), THEME["good"]),
            _stat("Sortino", _num(sortino, 3), THEME["cyan"]),
            _stat("MDD", _pct(max_drawdown), THEME["bad"]),
            _stat("Win", _pct(win_rate), THEME["warn"]),
        ],
        expand=True,
        equal=True,
    )
    info = Table.grid(expand=True)
    info.add_column()
    info.add_column(justify="right")
    info.add_row("Checkpoint Sharpe", _num(ckpt.get("sharpe"), 3) if isinstance(ckpt, dict) else "--")
    info.add_row("Checkpoint Sortino", _num(ckpt.get("sortino"), 3) if isinstance(ckpt, dict) else "--")
    info.add_row("Checkpoint MDD", _pct(ckpt.get("max_drawdown")) if isinstance(ckpt, dict) else "--")
    info.add_row("Profit Factor", _num(ckpt.get("profit_factor"), 3) if isinstance(ckpt, dict) else "--")
    return _panel(_t(lang, "performance_panel"), Group(cards, info), state, panel_id, THEME["good"])


def _per_regime_panel(latest_checkpoint: dict[str, Any] | None, state: DashboardState, panel_id: str, charset: str) -> Panel:
    metrics = latest_checkpoint.get("financialMetrics", {}) if latest_checkpoint else {}
    per_regime = metrics.get("per_regime", {}) if isinstance(metrics, dict) else {}
    if not per_regime:
        return _panel("Per-Regime", Text("No checkpoint", style=THEME["muted"]), state, panel_id, THEME["accent"])
    rows = Table.grid(expand=True)
    rows.add_column(width=10)
    rows.add_column(width=11, justify="right")
    rows.add_column(ratio=2)
    for label in REGIME_LABELS:
        payload = per_regime.get(label)
        if payload is None:
            continue
        mean_return = float(payload.get("mean_return", 0.0))
        rows.add_row(label.title(), _pct(mean_return), _bar(min(abs(mean_return) / 0.05, 1.0), REGIME_COLORS[label] if mean_return >= 0 else THEME["bad"], charset, 12))
    return _panel("Per-Regime", rows, state, panel_id, THEME["accent"])


def _runtime_panel(run: dict[str, Any], latest_metric: dict[str, Any] | None, state: DashboardState, panel_id: str, lang: str) -> Panel:
    runtime = run.get("runtime", {})
    cfg = run.get("config", {})
    rows = Table.grid(expand=True)
    rows.add_column()
    rows.add_column(justify="right")
    rows.add_row("Device", str(runtime.get("resolved_device", cfg.get("device", "--"))))
    rows.add_row("Torch", str(runtime.get("torch_version", "--")))
    rows.add_row("CUDA", str(runtime.get("cuda_version", "--")))
    rows.add_row("GPU", _gpu(runtime))
    rows.add_row("CPU Threads", str(runtime.get("cpu_threads", "--")))
    if latest_metric:
        rows.add_row("Global Step", f"{int(latest_metric.get('globalStep', 0)):,}")
    return _panel(_t(lang, "runtime"), rows, state, panel_id, THEME["cyan"])


def _config_panel(run: dict[str, Any], state: DashboardState, panel_id: str) -> Panel:
    cfg = run.get("config", {})
    rows = Table.grid(expand=True)
    rows.add_column()
    rows.add_column(justify="right")
    for key in ("experiment_name", "agent_type", "episodes", "episode_length", "seed", "fixed_eval_seeds", "hidden_dim", "n_experts", "gate_hidden_dim", "load_balance_weight", "curriculum_mode", "transaction_cost", "slippage_bps"):
        rows.add_row(key, str(cfg.get(key, "--")))
    return _panel("Config", rows, state, panel_id, THEME["accent"])


def _controls_panel(manager: TrainingManager, state: DashboardState, panel_id: str, lang: str) -> Panel:
    rows = Table.grid(expand=True)
    rows.add_column(width=14)
    rows.add_column(ratio=1)
    rows.add_row("1-5", "switch views")
    rows.add_row("Tab / Shift+Tab", "cycle focus")
    rows.add_row("Space", "resume" if manager.is_paused else "pause")
    rows.add_row("r", "toggle regime detail")
    rows.add_row("e", "toggle expert detail")
    rows.add_row("q / Esc", "exit")
    rows.add_row(_t(lang, "focus"), FOCUS[state.view_id][state.focus_index])
    return _panel(_t(lang, "controls"), rows, state, panel_id, THEME["warn"])


def _status_bar(run: dict[str, Any] | None, manager: TrainingManager, state: DashboardState, lang: str) -> Panel:
    snapshot = manager.live_snapshot()
    text = Text()
    text.append(f"{_t(lang, 'view')}: ", style=f"bold {THEME['text']}")
    text.append(_view_title(state.view_id, lang, "unicode"), style=THEME["accent_alt"])
    text.append("  |  ", style=THEME["muted"])
    text.append(f"{_t(lang, 'focus')}: ", style=f"bold {THEME['text']}")
    text.append(FOCUS[state.view_id][state.focus_index], style=THEME["cyan"])
    text.append("  |  ", style=THEME["muted"])
    text.append(f"{_t(lang, 'keys')}: 1-5 Tab Shift+Tab Space r e q", style=THEME["muted"])
    if run is not None:
        text.append("  |  ", style=THEME["muted"])
        text.append(_badge(_status_text(run, snapshot, lang), _status_color(run, snapshot)))
    return Panel(text, border_style=THEME["line"], box=ROUNDED, padding=(0, 1))


def _view_title(view_id: str, lang: str, charset: str) -> str:
    icon = {"overview": "🎯", "regime": "🔬", "expert": "⚡", "performance": "📈", "config": "🧩"}[view_id] if charset == "unicode" else f"[{VIEW_IDS.index(view_id) + 1}]"
    return f"{icon} {_t(lang, view_id)}"


def _panel(title: str, body: Any, state: DashboardState, panel_id: str, color: str) -> Panel:
    focused = FOCUS[state.view_id][state.focus_index] == panel_id
    edge = THEME["text"] if focused else color
    return Panel(body, title=Text(title, style=f"bold {edge}"), border_style=edge, box=ROUNDED, padding=(0, 1))


def _stat(title: str, value: str, color: str) -> Panel:
    return Panel(Text(value, style=f"bold {color}", justify="center"), title=title, border_style=color, box=ROUNDED, padding=(0, 1))


def _spark(values: list[float], palette: list[str], charset: str, width: int = 26) -> Text:
    if not values:
        return Text("n/a", style=THEME["muted"])
    chars = UNICODE_SPARK if charset == "unicode" else ASCII_SPARK
    sampled = values if len(values) <= width else [values[min(len(values) - 1, int(i * len(values) / width))] for i in range(width)]
    lo, hi = min(sampled), max(sampled)
    if math.isclose(lo, hi):
        return Text(chars[len(chars) // 2] * len(sampled), style=palette[len(palette) // 2])
    text = Text()
    for value in sampled:
        cidx = min(len(chars) - 1, max(0, int((value - lo) / (hi - lo + 1e-10) * (len(chars) - 1))))
        pidx = min(len(palette) - 1, max(0, int((value - lo) / (hi - lo + 1e-10) * (len(palette) - 1))))
        text.append(chars[cidx], style=palette[pidx])
    return text


def _bar(share: float, color: str, charset: str, width: int) -> Text:
    share = max(0.0, min(1.0, share))
    if charset == "unicode":
        filled = int(round(share * width))
        text = Text()
        text.append("█" * filled, style=color)
        text.append("·" * max(0, width - filled), style=THEME["line"])
        return text
    filled = int(round(share * width))
    text = Text()
    text.append("#" * filled, style=color)
    text.append("." * max(0, width - filled), style=THEME["line"])
    return text


def _empty_panel(message: str, title: str, color: str) -> Panel:
    return Panel(Group(Text(message, style=THEME["muted"]), Text(_t("en", "press_exit"), style=THEME["muted"])), title=title, border_style=color, box=ROUNDED)


def _badge(text: str, color: str) -> Text:
    return Text(f"「 {text} 」", style=f"bold {color}")


def _status_text(run: dict[str, Any], snapshot: dict[str, Any], lang: str) -> str:
    if snapshot.get("is_paused"):
        return _t(lang, "paused")
    status = str(run.get("status", "running")).lower()
    return _t(lang, {"running": "running", "completed": "done", "failed": "failed"}.get(status, "running"))


def _status_color(run: dict[str, Any], snapshot: dict[str, Any]) -> str:
    if snapshot.get("is_paused"):
        return THEME["warn"]
    status = str(run.get("status", "running")).lower()
    if status == "failed":
        return THEME["bad"]
    if status == "completed":
        return THEME["cyan"]
    return THEME["good"]


def _gpu(runtime: dict[str, Any]) -> str:
    if str(runtime.get("resolved_device", "cpu")).lower() != "cuda":
        return "cpu"
    now = time.time()
    if now - float(_GPU_CACHE["ts"]) < 2.0:
        return str(_GPU_CACHE["value"])
    value = "util n/a"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=0.4,
            check=False,
        )
        line = proc.stdout.strip().splitlines()[0]
        util, used, total = [chunk.strip() for chunk in line.split(",")[:3]]
        value = f"{util}% | {used}/{total} MiB"
    except Exception:
        value = "util n/a"
    _GPU_CACHE["ts"] = now
    _GPU_CACHE["value"] = value
    return value


def _get(mapping: dict[str, Any] | None, key: str, default: float | None = None) -> float | None:
    if mapping is None:
        return default
    value = mapping.get(key, default)
    if value is None:
        return default
    return float(value)


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{float(value) * 100:+.2f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes:02d}m {secs:02d}s"


def _value_color(value: float | None) -> str:
    if value is None:
        return THEME["text"]
    if value > 0:
        return THEME["good"]
    if value < 0:
        return THEME["bad"]
    return THEME["warn"]


def _lang(choice: str) -> str:
    if choice in {"en", "zh"}:
        return choice
    name = (locale.getdefaultlocale()[0] or "").lower()
    return "zh" if name.startswith("zh") else "en"


def _charset(choice: str) -> str:
    if choice in {"ascii", "unicode"}:
        return choice
    enc = (locale.getpreferredencoding(False) or "").lower()
    return "unicode" if "utf" in enc or "65001" in enc else "ascii"


def _t(lang: str, key: str) -> str:
    return I18N[lang].get(key, I18N["en"].get(key, key))


if __name__ == "__main__":
    main()
