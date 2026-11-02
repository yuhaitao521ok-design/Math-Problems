#!/usr/bin/env python3
"""Figures 2 and 3 of the Chinese notes: a real 2D Navier-Stokes simulation.

Pseudo-spectral vorticity formulation
    omega_t + u.grad omega = nu Laplacian omega,    u = curl^{-1} omega
on a 2pi-torus, with an integrating factor for diffusion and RK4 for the
advection (stable for the imaginary eigenvalues of the advection operator).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "/workspace/.fonts/NotoSansCJKsc-Regular.otf"
FONTB = "/workspace/.fonts/NotoSansCJKsc-Bold.otf"
for f in (FONT, FONTB):
    font_manager.fontManager.addfont(f)
NAME = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({
    "font.family": NAME, "font.size": 11, "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans", "savefig.bbox": "tight",
    "savefig.dpi": 200, "figure.dpi": 110,
})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", stem, flush=True)


N = 192
L = 2 * np.pi
nu = 2e-4
dt = 3e-3
TEND = 24.0
xs = np.arange(N) * L / N
X, Y = np.meshgrid(xs, xs, indexing="ij")
k = np.fft.fftfreq(N, d=1.0 / N)
KX, KY = np.meshgrid(k, k, indexing="ij")
K2 = KX ** 2 + KY ** 2
K2s = np.where(K2 == 0, 1.0, K2)
A = np.exp(-nu * K2 * dt)


def gauss(x0, y0, s):
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s ** 2))


omega = 2.2 * (gauss(np.pi - 0.85, np.pi, 0.30) + gauss(np.pi + 0.85, np.pi, 0.30)) \
    - 1.2 * gauss(np.pi, np.pi + 1.4, 0.28)
omega -= omega.mean()


def velocity(w):
    wh = np.fft.fftn(w)
    return (np.fft.ifftn(1j * KY * wh / K2s).real,
            np.fft.ifftn(-1j * KX * wh / K2s).real)


def nonlinear(w):
    u, v = velocity(w)
    wx = np.fft.ifftn(1j * KX * np.fft.fftn(w)).real
    wy = np.fft.ifftn(1j * KY * np.fft.fftn(w)).real
    return np.fft.fftn(-(u * wx + v * wy))


def step(w):
    wh = np.fft.fftn(w)
    k1 = nonlinear(w)
    k2 = nonlinear(np.fft.ifftn(A * (wh + 0.5 * dt * k1)).real)
    k3 = nonlinear(np.fft.ifftn(A * (wh + 0.5 * dt * k2)).real)
    k4 = nonlinear(np.fft.ifftn(A * (wh + dt * k3)).real)
    return np.fft.ifftn(A * (wh + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))).real


times = [0.0, 3.0, 8.0, 20.0]
snaps = {0.0: omega.copy()}
lp = {2: [], 4: [], 8: [], 16: []}
t_lp = []
nsteps = int(round(TEND / dt))
for n in range(nsteps + 1):
    t = n * dt
    if n % 100 == 0:
        t_lp.append(t)
        for p in lp:
            lp[p].append((np.abs(omega) ** p).mean() ** (1.0 / p))
    for tt in times[1:]:
        if tt not in snaps and t >= tt - dt / 2:
            snaps[tt] = omega.copy()
    omega = step(omega)

fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
vmax = np.abs(snaps[0.0]).max()
for ax, tt in zip(axes, times):
    im = ax.imshow(snaps[tt].T, origin="lower", extent=[0, L, 0, L],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set_title(f"$t={tt:g}$")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label=r"$\omega$")
save(fig, "fig02_2d_snapshots")

fig, ax = plt.subplots(figsize=(6.6, 3.7))
styles = {2: "-", 4: "--", 8: ":", 16: "-."}
for p in lp:
    arr = np.array(lp[p])
    ax.plot(t_lp, arr / arr[0], styles[p], lw=1.8, label=f"$p={p}$")
ymin = min((np.array(lp[p]) / np.array(lp[p])[0]).min() for p in lp)
ax.set_ylim(0.995 * ymin, 1.002)
ax.set_xlabel("时间 $t$")
ax.set_ylabel(r"$\|\omega(t)\|_p \,/\, \|\omega(0)\|_p$")
ax.set_title("二维涡量的 $L^p$ 范数随时间不增（实算）")
ax.legend(frameon=False, ncol=4, loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
ax.annotate(r"$\partial_t\omega+u\cdot\nabla\omega=\nu\Delta\omega$" + "\n输运项积分为零，只剩耗散",
            xy=(0.30, 0.30), xycoords="axes fraction", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))
save(fig, "fig03_2d_norms")

print("max|omega|:", [round(float(np.abs(snaps[t]).max()), 3) for t in times])
print("||omega||_2 ratio at end:", round(lp[2][-1] / lp[2][0], 4))
print("finite:", np.isfinite(omega).all(), " max speed:",
      round(float(np.abs(velocity(omega)[0]).max() + np.abs(velocity(omega)[1]).max()), 3))
