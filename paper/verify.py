#!/usr/bin/env python3
"""Numerical verification of the identities used in ns-forced-vs-unforced.tex.

All fields are trigonometric polynomials (or rapidly decaying profiles on a
large periodic box), so every test is an exact identity up to round-off.

T1  Prescription theorem (Thm 7.1): div v = 0, residual = 0, div f = 0
    (the last tests the sign convention in the pressure equation).
T2  Scaling (Prop 2.1): R(u_l,p_l,f_l)(x) = l^3 R(u,p,f)(l x).
T3  Self-similar price law (Thm 8.1): for a Gaussian profile ||F||_inf > 0 and
    the self-similar field with force l^3 F(l x) has zero residual.
T4  Two-dimensional vorticity cancellation (Lemma 5.2).
"""
import numpy as np

nu = 1.0
N = 48
L = 2.0 * np.pi
x = np.arange(N) * L / N
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
kk = np.fft.fftfreq(N, d=1.0 / N)
KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
K2 = KX ** 2 + KY ** 2 + KZ ** 2
K2s = np.where(K2 == 0, 1.0, K2)


def fft(f):
    return np.fft.fftn(f)


def ifft(F):
    return np.fft.ifftn(F).real


def grad(f):
    F = fft(f)
    return (ifft(1j * KX * F), ifft(1j * KY * F), ifft(1j * KZ * F))


def div(u):
    return sum(grad(u[i])[i] for i in range(3))


def lap(f):
    return ifft(-K2 * fft(f))


def conv(u, v):
    return tuple(sum(u[j] * grad(v[i])[j] for j in range(3)) for i in range(3))


def curl(a):
    d1, d2, d3 = grad(a[0]), grad(a[1]), grad(a[2])
    return (d2[2] - d3[1], d3[0] - d1[2], d1[1] - d2[0])


def trig(seed, kmax=2):
    rng = np.random.default_rng(seed)
    f = np.zeros_like(X)
    for _ in range(6):
        kx, ky, kz = rng.integers(-kmax, kmax + 1, size=3)
        ph = rng.uniform(0, 2 * np.pi)
        f += rng.normal() * np.cos(kx * X + ky * Y + kz * Z + ph)
    return f


def pressure(g):
    return ifft(fft(g) / K2s)


def residual(u, p, frc):
    w = conv(u, u)
    gp = grad(p)
    return tuple(w[i] - nu * lap(u[i]) + gp[i] - frc[i] for i in range(3))


idx = np.fft.fftfreq(N, d=1.0 / N).astype(int)
idx = np.where(idx < N // 2, idx, idx - N)


def stretch(g, lam, power, n=N, idxn=None):
    """return lam^power * g(lam x) by spectral lookup (lam a positive integer)."""
    if idxn is None:
        idxn = idx
    gh = fft(g) if n == N else np.fft.fftn(g)
    out = np.zeros_like(gh)
    sel = [(i, k) for i, k in enumerate(idxn) if k % lam == 0]
    for i, ki in sel:
        for j, kj in sel:
            for m, km in sel:
                out[i, j, m] = gh[ki // lam % n, kj // lam % n, km // lam % n]
    res = (ifft(out) if n == N else np.fft.ifftn(out).real)
    return res * lam ** power


print("=" * 68)
print("T1  prescription theorem: div v = 0, residual = 0, div f = 0")
print("=" * 68)
A1 = (trig(1), trig(2), trig(3))
A2 = (trig(4), trig(5), trig(6))
U1, U2 = curl(A1), curl(A2)
t0 = 0.7
a = lambda t: 1.0 + 0.4 * np.sin(2 * t)
b = lambda t: 0.6 * np.cos(t)
da = lambda t: 0.8 * np.cos(2 * t)
db = lambda t: -0.6 * np.sin(t)
v = tuple(a(t0) * U1[i] + b(t0) * U2[i] for i in range(3))
dv = tuple(da(t0) * U1[i] + db(t0) * U2[i] for i in range(3))
w = conv(v, v)
q = pressure(div(w))
gq = grad(q)
f = tuple(dv[i] + w[i] - nu * lap(v[i]) + gq[i] for i in range(3))
res = tuple(dv[i] + w[i] - nu * lap(v[i]) + gq[i] - f[i] for i in range(3))
print(f"  max |div v|    = {np.abs(div(v)).max():.3e}")
print(f"  max |residual| = {max(np.abs(r).max() for r in res):.3e}")
print(f"  max |div f|    = {np.abs(div(f)).max():.3e}   <-- pressure sign test")
print(f"  sup |v|        = {max(np.abs(vv).max() for vv in v):.3f}")

print()
print("=" * 68)
print("T2  scaling: R(u_l,p_l,f_l) = l^3 R(u,p,f)(l x),  l = 2, 3")
print("=" * 68)
u = (trig(11, 3), trig(12, 3), trig(13, 3))
p = trig(14, 3)
frc = (trig(15, 3), trig(16, 3), trig(17, 3))
R = residual(u, p, frc)
Rmax = max(np.abs(r).max() for r in R)
for lam in (2, 3):
    ul = tuple(stretch(u[i], lam, 1) for i in range(3))
    pl = stretch(p, lam, 2)
    fl = tuple(stretch(frc[i], lam, 3) for i in range(3))
    Rl = residual(ul, pl, fl)
    Rt = tuple(stretch(R[i], lam, 3) for i in range(3))
    err = max(np.abs(Rl[i] - Rt[i]).max() for i in range(3))
    print(f"  lambda = {lam}: sup error = {err:.3e}   (sup |R| = {Rmax:.3e})")

print()
print("=" * 68)
print("T3  self-similar price law (Thm 8.1), Gaussian profile")
print("=" * 68)
M = 96
Lb = 32.0
xb = np.arange(M) * Lb / M - Lb / 2
Xb, Yb, Zb = np.meshgrid(xb, xb, xb, indexing="ij")
kb = 2 * np.pi * np.fft.fftfreq(M, d=Lb / M)
KXb, KYb, KZb = np.meshgrid(kb, kb, kb, indexing="ij")
K2b = KXb ** 2 + KYb ** 2 + KZb ** 2
K2bs = np.where(K2b == 0, 1.0, K2b)


def grad_b(f):
    F = np.fft.fftn(f)
    return (
        np.fft.ifftn(1j * KXb * F).real,
        np.fft.ifftn(1j * KYb * F).real,
        np.fft.ifftn(1j * KZb * F).real,
    )


def lap_b(f):
    return np.fft.ifftn(-K2b * np.fft.fftn(f)).real


def conv_b(u, v):
    return tuple(sum(u[j] * grad_b(v[i])[j] for j in range(3)) for i in range(3))


psi = np.exp(-(Xb ** 2 + Yb ** 2 + Zb ** 2) / 2)
dpsi = grad_b(psi)
U = (dpsi[1], -dpsi[0], 0.0 * psi)          # curl(psi e_z) = (-y, x, 0) exp(-r^2/2)
print(f"  sup |div U|   = {np.abs(sum(grad_b(U[i])[i] for i in range(3))).max():.3e}")
ygrad = tuple(
    Xb * grad_b(U[i])[0] + Yb * grad_b(U[i])[1] + Zb * grad_b(U[i])[2]
    for i in range(3)
)
G = tuple(U[i] + ygrad[i] for i in range(3))   # U + y.grad U
wb = conv_b(U, U)
Q = np.fft.ifftn(np.fft.fftn(sum(grad_b(wb[i])[i] for i in range(3))) / K2bs).real
gQ = grad_b(Q)
F = tuple(G[i] + wb[i] - nu * lap_b(U[i]) + gQ[i] for i in range(3))
Fmax = max(np.abs(Fi).max() for Fi in F)
print(f"  sup |F|_inf   = {Fmax:.6e}   (must be > 0)")


# chain rule: v(x,t) = lambda U(lambda x), lambda(t0) = 1 for T = 1, a = 1
def Uclosed(y1, y2, y3):
    r2 = y1 ** 2 + y2 ** 2 + y3 ** 2
    return (-y2 * np.exp(-r2 / 2), y1 * np.exp(-r2 / 2), 0.0 * y1)


T0, acc = 1.0, 1.0
lam = lambda t: (2 * acc * (T0 - t)) ** -0.5
t0 = T0 - 1.0 / (2 * acc)
h = 1e-3


def vfield(t):
    l = lam(t)
    return tuple(l * c for c in Uclosed(l * Xb, l * Yb, l * Zb))


vm = vfield(t0 - h)
vp = vfield(t0 + h)
dvdt = tuple((vp[i] - vm[i]) / (2 * h) for i in range(3))
print(f"  lambda(t0) = {lam(t0):.12f},   dot lambda / lambda^3 = "
      f"{(lam(t0 + h) - lam(t0 - h)) / (2 * h) / lam(t0) ** 3:.12f}")
err = max(np.abs(dvdt[i] - G[i]).max() for i in range(3))
print(f"  chain rule at lambda = 1: sup |dv/dt - (U + x.grad U)| = {err:.3e}")
print("  price law  (T-t)^(3/2) ||f(t)||_inf = (2a)^(-3/2) ||F||_inf:")
for tt in (0.25, 0.5, 0.875):
    lt = lam(tt)
    print(f"    t = {tt:5.3f}  lambda = {lt:7.4f}  "
          f"(T-t)^(3/2) lambda^3 ||F|| = {(T0 - tt) ** 1.5 * lt ** 3 * Fmax:.6f}  "
          f"(2a)^(-3/2) ||F|| = {(2 * acc) ** -1.5 * Fmax:.6f}")

print()
print("=" * 68)
print("T4  2D vorticity cancellation: int (u.grad w) w dx = 0")
print("=" * 68)
N2 = 64
x2 = np.arange(N2) * 2 * np.pi / N2
X2, Y2 = np.meshgrid(x2, x2, indexing="ij")
k2 = np.fft.fftfreq(N2, d=1.0 / N2)
KX2, KY2 = np.meshgrid(k2, k2, indexing="ij")


def grad2(f):
    F = np.fft.fftn(f)
    return (np.fft.ifftn(1j * KX2 * F).real, np.fft.ifftn(1j * KY2 * F).real)


rng = np.random.default_rng(7)
psi2 = sum(
    rng.normal()
    * np.cos(
        rng.integers(1, 4) * X2 + rng.integers(1, 4) * Y2 + rng.uniform(0, 6.3)
    )
    for _ in range(5)
)
u2 = (grad2(psi2)[1], -grad2(psi2)[0])
w2 = grad2(u2[1])[0] - grad2(u2[0])[1]
wx, wy = grad2(w2)
adv = u2[0] * wx + u2[1] * wy
cell = (2 * np.pi / N2) ** 2
num = (adv * w2).sum() * cell
den_u = (np.abs(u2[0]).max() + np.abs(u2[1]).max())
den_g = np.sqrt((wx ** 2 + wy ** 2).sum() * cell)
den_w = np.sqrt((w2 ** 2).sum() * cell)
print(f"  |int (u.grad w) w| = {abs(num):.3e}")
print(f"  normalisation     = {den_u * den_g * den_w:.3e}")
print(f"  relative          = {abs(num) / (den_u * den_g * den_w):.3e}")

