#!/usr/bin/env python3
"""Symbolic (sympy) re-derivation of the identities used in the paper.

Every check recomputes the identity from its definition, independently of the
paper's derivation.  Output is used verbatim in the cross-validation report.
"""
import sympy as sp

x, y, z, t, T, a, nu, X, Y, Z = sp.symbols("x y z t T a nu X Y Z", real=True)
q = sp.symbols("q", positive=True)
a = sp.symbols("a", positive=True)
T = sp.symbols("T", positive=True)
t = sp.symbols("t", positive=True)
lam = (2 * a * (T - t)) ** sp.Rational(-1, 2)
lamdot = sp.simplify(sp.diff(lam, t))

print("=" * 72)
print("C1  chain rule  d/dt[lam*U(lam*x)] = lamdot*(U + y U'),  y = lam*x")
for U in (sp.exp(-Y ** 2 / 2), Y ** 3 - Y, sp.cos(Y)):
    lhs = sp.diff(lam * U.subs(Y, lam * X), t)
    yv = lam * X
    rhs = lamdot * (U.subs(Y, yv) + yv * sp.diff(U, Y).subs(Y, yv))
    print("   U =", U, " ->  difference =", sp.simplify(lhs - rhs))
print("   lamdot / lam^3 =", sp.simplify(lamdot / lam ** 3), "(must be a)")

print()
print("=" * 72)
print("C2  exponent bookkeeping  lam^(3-3/q) = (2a(T-t))^(-(3/2)(1-1/q))")
lhs = sp.expand_log(sp.log(lam ** (3 - 3 / q)), force=True)
rhs = sp.expand_log(sp.log((2 * a * (T - t)) ** (-sp.Rational(3, 2) * (1 - 1 / q))), force=True)
print("   log-difference =", sp.simplify(lhs - rhs), "(must be 0)")

print()
print("=" * 72)
print("C3  Kato-type inequality  Delta|w|^p - p|w|^(p-2) w Delta w = p(p-1)|w|^(p-2)|grad w|^2")
w = sp.sin(x) * sp.cos(y)
lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
gw2 = sp.diff(w, x) ** 2 + sp.diff(w, y) ** 2
for p in (2, 4):
    resid = sp.simplify(lap(sp.Abs(w) ** p) - p * sp.Abs(w) ** (p - 2) * w * lap(w)
                        - p * (p - 1) * sp.Abs(w) ** (p - 2) * gw2)
    print(f"   p={p}: residual =", resid, "(must be 0)")

print()
print("=" * 72)
print("C4  Prodi-Serrin / Young exponents")
print("   1/q + (q-2)/(2q) + 1/2      =", sp.simplify(1 / q + (q - 2) / (2 * q) + sp.Rational(1, 2)))
print("   (1-3/q)*(2q/(q-3))          =", sp.simplify((1 - 3 / q) * 2 * q / (q - 3)))
print("   1/sigma + 1/sigma' , sigma=2q/(q+3), sigma'=2q/(q-3)  =",
      sp.simplify((q + 3) / (2 * q) + (q - 3) / (2 * q)))

print()
print("=" * 72)
print("C5  vorticity of the Gaussian profile used in Cor. 7.3 / Thm 8.1")
r2 = x ** 2 + y ** 2
u1 = -y * sp.exp(-r2 / 2)
u2 = x * sp.exp(-r2 / 2)
w2d = sp.simplify(sp.diff(u2, x) - sp.diff(u1, y))
print("   curl u =", w2d, " (should be (2 - r^2) exp(-r^2/2))")
print("   check  :", sp.simplify(w2d - (2 - r2) * sp.exp(-r2 / 2)))

print()
print("=" * 72)
print("C6  2D vorticity identity  (u.grad)w = curl((u.grad)u)  for div-free u")
psi = sp.sin(x) * sp.sin(y)
uu1, uu2 = sp.diff(psi, y), -sp.diff(psi, x)
ww = sp.simplify(sp.diff(uu2, x) - sp.diff(uu1, y))
adv = sp.simplify(uu1 * sp.diff(ww, x) + uu2 * sp.diff(ww, y))
curlconv = sp.simplify(sp.diff(uu1 * sp.diff(uu2, x) + uu2 * sp.diff(uu2, y), x)
                      - sp.diff(uu1 * sp.diff(uu1, x) + uu2 * sp.diff(uu1, y), y))
print("   div u =", sp.simplify(sp.diff(uu1, x) + sp.diff(uu2, y)))
print("   (u.grad)w - curl((u.grad)u) =", sp.simplify(adv - curlconv))
