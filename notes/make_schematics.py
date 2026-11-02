#!/usr/bin/env python3
"""Schematic figures for the Chinese lecture notes (vector PDF + PNG preview)."""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Rectangle, Arc

FONT = "/workspace/.fonts/NotoSansCJKsc-Regular.otf"
FONTB = "/workspace/.fonts/NotoSansCJKsc-Bold.otf"
for f in (FONT, FONTB):
    font_manager.fontManager.addfont(f)
NAME = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({
    "font.family": NAME,
    "font.size": 11,
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "figure.dpi": 110,
})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
BLUE, RED, GREEN, GREY, ORANGE = "#2b6cb0", "#c53030", "#2f855a", "#4a5568", "#dd6b20"


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", stem)


def box(ax, xy, w, h, text, fc="#ebf4ff", ec=BLUE, fs=10, tc="black", lw=1.2, r=0.02):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.5)


def arrow(ax, p0, p1, color=GREY, style="-|>", lw=1.4, ls="-", rad=0.0, z=1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- fig01 scaling
lam = np.logspace(-0.7, 1.1, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
a1.loglog(lam, lam ** 0, lw=2, color=BLUE, label=r"$\|u_\lambda\|_{L^3}$（尺度不变）")
a1.loglog(lam, lam ** 0.5, lw=2, color=GREEN, label=r"$\|\nabla u_\lambda\|_{L^2}\propto\lambda^{1/2}$")
a1.loglog(lam, lam ** 3, lw=2, color=RED, label=r"$\|f_\lambda\|_{L^\infty}\propto\lambda^{3}$")
a1.axvline(1.0, color=GREY, lw=0.8, ls=":")
a1.set_xlabel(r"放大倍数 $\lambda$")
a1.set_ylabel("范数（归一到 $\\lambda=1$）")
a1.set_title("同一解在不同尺度下的“大小”")
a1.legend(frameon=False, fontsize=9, loc="upper left")
a1.grid(alpha=0.25, which="both")

a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.axis("off")
a2.add_patch(Rectangle((0.03, 0.22), 0.38, 0.50, fc="#f7fafc", ec=GREY, lw=1.2))
a2.add_patch(Ellipse((0.22, 0.47), 0.13, 0.10, fc="#bee3f8", ec=BLUE, lw=1.3))
a2.text(0.22, 0.15, "尺度 $L$：整块区域", ha="center", fontsize=8.8)
a2.add_patch(Rectangle((0.175, 0.425), 0.09, 0.09, fill=False, ec=RED, ls="--", lw=1.1))
arrow(a2, (0.45, 0.47), (0.53, 0.47), color=RED, lw=1.3)
a2.text(0.49, 0.51, "放大", ha="center", fontsize=8, color=RED)
a2.add_patch(Rectangle((0.57, 0.22), 0.38, 0.50, fc="#fff5f5", ec=RED, lw=1.2))
a2.add_patch(Ellipse((0.76, 0.47), 0.24, 0.19, fc="#bee3f8", ec=BLUE, lw=1.5))
a2.text(0.76, 0.15, "尺度 $L/\\lambda$：同一个涡", ha="center", fontsize=8.8, color=RED)
a2.text(0.985, 0.99,
        r"$u_\lambda=\lambda\,u(\lambda x,\lambda^2 t)$" + "\n"
        r"$p_\lambda=\lambda^2p$   $f_\lambda=\lambda^3f$",
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
a2.set_title("“尺度不变”到底在说什么")
save(fig, "fig01_scaling")

# ------------------------------------------------------------- fig04 stretching
fig, ax = plt.subplots(figsize=(9.6, 3.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
t = np.linspace(0, 2 * np.pi, 200)
# tube surface: radius r(x) is smallest in the middle (the tube is being stretched)
xs = np.linspace(1.15, 8.85, 400)
rr = 0.30 + 0.30 * ((xs - 5.0) / 3.85) ** 2
ax.fill_between(xs, 2.0 - rr, 2.0 + rr, color="#bee3f8", alpha=0.55, lw=0)
ax.plot(xs, 2.0 + rr, color=BLUE, lw=1.5)
ax.plot(xs, 2.0 - rr, color=BLUE, lw=1.5)
for xc in (1.6, 5.0, 8.4):
    r = 0.30 + 0.30 * ((xc - 5.0) / 3.85) ** 2
    ax.plot(xc + 0.12 * np.cos(t), 2.0 + r * np.sin(t), color=BLUE, lw=1.0, ls="--")
for x0, dx in ((1.15, -0.55), (8.85, 0.55)):
    arrow(ax, (x0, 2.0), (x0 + dx, 2.0), color=RED, lw=2.2)
ax.text(0.35, 2.30, "拉伸", color=RED, fontsize=10)
ax.text(8.55, 2.30, "拉伸", color=RED, fontsize=10)
ax.add_patch(Arc((5.0, 2.0), 0.62, 0.62, theta1=30, theta2=330, color=BLUE, lw=1.4))
ax.annotate("", xy=(5.24, 2.20), xytext=(5.20, 2.10),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3))
ax.text(5.0, 2.0, r"$\omega$", ha="center", va="center", fontsize=11, color=BLUE)
ax.text(0.2, 3.55,
        "涡管被拉长 → 截面变小 → 涡量变强（涡管定理：环量守恒）",
        fontsize=10.5, color=GREY)
ax.text(0.2, 0.75,
        r"三维：$\partial_t\omega+(u\cdot\nabla)\omega=(\omega\cdot\nabla)u+\nu\Delta\omega$"
        + "\n拉伸项 $(\\omega\\cdot\\nabla)u$ 可正可负，$\\|\\omega\\|_p$ 不再单调",
        fontsize=10, va="bottom", color="black",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fffaf0", ec=ORANGE))
ax.set_title("三维的麻烦：涡管拉伸（vortex stretching）")
save(fig, "fig04_stretching")

# --------------------------------------------------------------- fig13 blow-up
fig, ax = plt.subplots(figsize=(6.6, 3.7))
t = np.linspace(0, 1, 400)
ax.plot(t, 1 + 0.6 * np.sin(3 * t), lw=2.2, color=GREEN, label="整体正则：梯度始终有界")
tb = 0.72
ax.plot(t[t < tb], 1 + 0.6 * np.sin(3 * t[t < tb]) + 0.25 / (tb - t[t < tb]) ** 0.55,
        lw=2.2, color=RED, label="有限时间爆破：梯度发散")
ax.axvline(tb, color=RED, ls="--", lw=1)
ax.text(tb + 0.02, 3.3, r"$T^*$", color=RED, fontsize=12)
ax.set_ylim(0, 6.5)
ax.set_xlabel("时间 $t$")
ax.set_ylabel(r"$\|\nabla u(t)\|_\infty$（示意）")
ax.set_title("“爆破”是什么意思（示意图，非真实数据）")
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
ax.grid(alpha=0.25)
ax.annotate("BKM 判据：\n$\\int_0^{T^*}\\|\\nabla u\\|_\\infty\\,dt=\\infty$",
            xy=(0.55, 4.4), fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))
save(fig, "fig13_blowup")

# ------------------------------------------------------------ fig05 prescription
fig, ax = plt.subplots(figsize=(10.4, 4.6))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, (0.30, 3.35), 2.55, 0.85, "任意光滑、无散的\n路径 $v(x,t)$", fc="#ebf4ff")
box(ax, (3.45, 3.35), 2.75, 0.85, "解压力、构造外力\n$q,\\ f$", fc="#fff5f5", ec=RED)
box(ax, (6.75, 3.35), 3.30, 0.85, "$(v,q,f)$ 就是\n纳维–斯托克斯的精确解", fc="#f0fff4", ec=GREEN)
arrow(ax, (2.90, 3.775), (3.40, 3.775))
arrow(ax, (6.25, 3.775), (6.70, 3.775))
ax.text(5.0, 2.80,
        r"$-\Delta q=\mathrm{div}((v\cdot\nabla)v)$" + "\n"
        r"$f=\partial_t v+(v\cdot\nabla)v-\nu\Delta v+\nabla q$",
        ha="center", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))
ax.text(0.30, 2.35, "带外力（选项 C）：没有存在性问题，只有定量问题", fontsize=11, color=RED)
box(ax, (0.30, 0.55), 2.55, 0.95, "无外力\n$f\\equiv0$", fc="#f7fafc", ec=GREY)
box(ax, (3.45, 0.55), 2.75, 0.95, "约束：必须满足\n$\\partial_t v+\\dots+\\nabla q=0$", fc="#f7fafc", ec=GREY)
box(ax, (6.75, 0.55), 3.30, 0.95, "只允许“刚刚好”的\n一条路径 —— 未知！", fc="#fffaf0", ec=ORANGE)
arrow(ax, (2.90, 1.025), (3.40, 1.025), color=GREY)
arrow(ax, (6.25, 1.025), (6.70, 1.025), color=GREY)
ax.set_title("指定力定理：自由的外力把“解方程”变成“选路径”")
# ------------------------------------------------------------ fig05 prescription
save(fig, "fig05_prescription")

# ------------------------------------------------------------ fig06 manufactured
T = 1.0
tt = np.linspace(0, 0.96, 400)
lam = (2 * (T - tt)) ** -0.5
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 3.8))
a1.plot(tt, lam, lw=2.2, color=BLUE)
a1.axvline(T, color=GREY, ls=":", lw=1)
a1.set_xlabel("时间 $t$"); a1.set_ylabel(r"$\lambda(t)=(2a(T-t))^{-1/2}$")
a1.set_title("自相似放大因子在 $T$ 处发散")
a1.grid(alpha=0.25)
a1.text(0.05, 6.5, r"$\lambda(t)\to\infty$ 当 $t\to T$", fontsize=10)

for s, c, ls in ((1.0, RED, "-"), (1.5, ORANGE, "--"), (2.0, GREEN, ":")):
    a2.semilogy(tt, lam ** (s - 0.5), ls, color=c, lw=2, label=f"$s={s}$")
a2.set_xlabel("时间 $t$"); a2.set_ylabel(r"$\|v(t)\|_{H^s}$（对数轴）")
a2.set_title("$H^s$ 范数爆破（$s>1/2$），可在任意指定 $T$ 发生")
a2.legend(frameon=False, fontsize=9)
a2.grid(alpha=0.25, which="both")
save(fig, "fig06_manufactured")

# -------------------------------------------------------------------- fig07 price
tau = np.logspace(-3, 0.3, 300)
c = 1.153384
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.loglog(tau, c * tau ** -1.5, lw=2.2, color=RED, label=r"$\|f(t)\|_\infty=(2a)^{-3/2}\|F\|_\infty\,\tau^{-3/2}$")
pts = np.array([1.0, 0.1, 0.01])
ax.loglog(pts, c * pts ** -1.5, "o", color="black", ms=6, label="代价定律上的取值")
ax.loglog(tau, 1.0 * tau ** -1.0, ls="--", lw=1.2, color=GREY, label=r"参考斜率 $-1$")
ax.set_xlabel(r"距爆破的时间 $\tau=T-t$")
ax.set_ylabel(r"$\|f(t)\|_\infty$")
ax.set_title("外力的“价目表”：必须比 $1/\\tau$ 更快地发散")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(alpha=0.25, which="both")
ax.annotate("斜率 $-3/2$\n（自相似爆破）", xy=(0.06, c * 0.06 ** -1.5),
            xytext=(0.004, 120), fontsize=9.5,
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1))
save(fig, "fig07_price")

# ----------------------------------------------------------------- fig08 timeline
events = [
    ("1934", "Leray\n弱解存在"),
    ("1951", "Hopf\n能量不等式"),
    ("1959/62", "Prodi–Serrin\n条件正则"),
    ("1982", "CKN\n部分正则"),
    ("1984", "BKM\n爆破判据"),
    ("2000", "Clay\n百万美元"),
    ("2026.9", "声称：带外力\n有限时间爆破"),
    ("2026.9.17", "CIV：外力\n不能解析"),
    ("2026.10", "721 篇手稿\n373 个成果族"),
]
fig, ax = plt.subplots(figsize=(12.6, 3.6))
ax.set_xlim(-0.6, len(events) - 0.4); ax.set_ylim(-1.35, 1.35); ax.axis("off")
ax.plot([-0.4, len(events) - 0.6], [0, 0], color=GREY, lw=2)
for i, (yr, txt) in enumerate(events):
    up = i % 2 == 0
    y = 0.55 if up else -0.55
    c = RED if yr.startswith("2026") else BLUE
    ax.plot([i, i], [0, y * 0.55], color=c, lw=1)
    ax.plot(i, 0, "o", color=c, ms=6)
    ax.text(i, y, txt, ha="center", va="center", fontsize=8.8, color=c,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, lw=0.8))
    ax.text(i - 0.10, 0.13 if up else -0.13, yr, ha="right",
            va="bottom" if up else "top", fontsize=8, color=GREY)
ax.set_title("九十年：从“弱解存在”到“带外力爆破”")
save(fig, "fig08_timeline")

# ------------------------------------------------------------- fig09 verification
labels = ["T1  $\\mathrm{div}\\,v$", "T1  $\\mathrm{div}\\,f$", "T2  $\\lambda=2$",
          "T2  $\\lambda=3$", "T3  链式法则", "T4  涡量恒等式"]
vals = [-12.68, -9.82, -11.35, -11.25, -5.58, -14.40]
fig, ax = plt.subplots(figsize=(7.4, 3.6))
cols = [GREEN, GREEN, BLUE, BLUE, ORANGE, GREEN]
ax.barh(range(len(vals)), vals, color=cols, height=0.6)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("$\\log_{10}$（误差）")
ax.set_xlim(-16, 0)
ax.axvline(-6, color=GREY, ls="--", lw=1)
ax.set_ylim(6.05, -0.5)
ax.text(-6.05, 5.85, "虚线：有限差分截断误差量级", fontsize=8, color=GREY, ha="center", va="top")
for i, v in enumerate(vals):
    ax.text(v - 0.15, i, f"{v:.1f}", va="center", ha="right", fontsize=8.5)
ax.set_title("论文附录 B 的四项数值核验（越靠左越好）")
ax.grid(axis="x", alpha=0.25)
save(fig, "fig09_verification")

# -------------------------------------------------------------------- fig10 clay
items = [
    ("庞加莱猜想", "已解决（Perelman, 2002–03）", GREEN),
    ("黎曼猜想", "开放", RED),
    ("P vs NP", "开放", RED),
    ("纳维–斯托克斯", "开放；2026 年只声称解决“带外力”变体", ORANGE),
    ("杨–米尔斯", "开放", RED),
    ("霍奇猜想", "开放（仅特殊情形有结果）", RED),
    ("BSD 猜想", "开放（仅低秩情形有结果）", RED),
]
fig, ax = plt.subplots(figsize=(9.4, 3.9))
ax.set_xlim(0, 10); ax.set_ylim(0, len(items) + 0.4); ax.axis("off")
for i, (name, status, col) in enumerate(items):
    y = len(items) - i - 0.5
    ax.add_patch(FancyBboxPatch((0.3, y - 0.32), 2.9, 0.64,
                                boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc="#f7fafc", ec="0.6"))
    ax.text(1.75, y, name, ha="center", va="center", fontsize=10)
    ax.add_patch(FancyBboxPatch((3.4, y - 0.32), 6.2, 0.64,
                                boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=col, ec=col, alpha=0.14))
    ax.text(3.65, y, status, ha="left", va="center", fontsize=9.5, color=col)
ax.set_title("Clay 千禧七题：七题只落地一题（2026 年 11 月）")
save(fig, "fig10_clay")

# -------------------------------------------------------------------- fig11 lean
fig, ax = plt.subplots(figsize=(10.6, 3.9))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.9); ax.axis("off")
boxes = [
    ("内部模型\n手稿（166 页）", 0.25, 2.55, 1.85, "#ebf4ff", BLUE),
    ("自然语言\n同行评审", 2.55, 2.55, 1.75, "#f7fafc", GREY),
    ("Lean\n形式化", 4.75, 2.55, 1.55, "#fff5f5", RED),
    ("Clay 评审\n（刻意不赶时间）", 6.75, 2.55, 2.15, "#f7fafc", GREY),
    ("奖金", 9.30, 2.55, 1.05, "#f0fff4", GREEN),
]
for txt, x, y, w, fc, ec in boxes:
    box(ax, (x, y), w, 0.85, txt, fc=fc, ec=ec, fs=9.2)
for x0, x1 in ((2.10, 2.52), (4.30, 4.72), (6.30, 6.72), (8.90, 9.27)):
    arrow(ax, (x0, 2.975), (x1, 2.975))
ax.plot([1.15, 1.15, 5.50, 5.50], [2.52, 2.08, 2.08, 2.55], ls="--", color=RED, lw=1.3)
arrow(ax, (5.50, 2.30), (5.50, 2.52), color=RED, lw=1.3)
ax.text(3.35, 1.25, "翻译可能不忠实：语义忠实的自动形式化\n难度高于停机问题（SCI $=\\infty$）",
        ha="center", fontsize=9.5, color=RED,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec=RED))
ax.text(0.25, 0.55, "注：Lean 验证的是“某个形式命题有证明”，不是“论文里的那个命题有证明”。",
        fontsize=9.5, color=GREY)
ax.set_title("一篇声称证明的“证据链”：形式化不能替代审稿")
save(fig, "fig11_lean")

# --------------------------------------------------------------- fig12 2d vs 3d
fig, ax = plt.subplots(figsize=(10.4, 4.3))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.3); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.3, 0.3), 4.7, 3.7,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="#f0fff4", ec=GREEN, lw=1.3))
ax.add_patch(FancyBboxPatch((5.4, 0.3), 4.7, 3.7,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="#fff5f5", ec=RED, lw=1.3))
ax.text(2.65, 3.62, "二维：可以证明", ha="center", fontsize=12, color=GREEN)
ax.text(7.75, 3.62, "三维：仍然开放", ha="center", fontsize=12, color=RED)
ax.text(0.55, 3.12, r"$\partial_t\omega+u\cdot\nabla\omega=\nu\Delta\omega$",
        fontsize=11, va="top")
ax.text(0.55, 2.45, "· 涡量只被输运、被扩散\n· 输运项积分为零\n· $L^p$ 范数不增\n· 全局光滑解（定理 5.1）",
        fontsize=9.8, va="top", linespacing=1.6)
ax.text(5.65, 3.12, r"$\partial_t\omega+(u\cdot\nabla)\omega=(\omega\cdot\nabla)u+\nu\Delta\omega$",
        fontsize=11, va="top")
ax.text(5.65, 2.45, "· 多出拉伸项 $(\\omega\\cdot\\nabla)u$\n· 该项无符号、可放大涡量\n· 没有任何单调量\n· 全局正则 = 千禧问题",
        fontsize=9.8, va="top", linespacing=1.6)
ax.text(0.55, 0.55, "同一套能量估计，在二维足够", fontsize=9, color=GREY)
ax.text(5.65, 0.55, "同一套能量估计，在三维差一个临界量", fontsize=9, color=GREY)
ax.set_title("为什么二维能做完、三维做不到：差别只有一项")
save(fig, "fig12_2d_vs_3d")


