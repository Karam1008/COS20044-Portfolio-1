"""Week 3 artefact: physics primer - waves, superposition, wave packets, uncertainty."""
import numpy as np
from style import plt, save

print("=" * 66)
print("WEEK 3  |  PHYSICS PRIMER")
print("=" * 66)

# ======================================================================
# 1. Classical wave behaviour   y(x,t) = A cos(kx - wt + phi)
# ======================================================================
A, k, w = 1.0, 2 * np.pi / 4.0, 2 * np.pi / 2.0    # lambda = 4 m, T = 2 s
lam, T, f = 2 * np.pi / k, 2 * np.pi / w, w / (2 * np.pi)
print(f"1  A = {A}, k = {k:.4f} rad/m -> lambda = {lam:.2f} m")
print(f"   omega = {w:.4f} rad/s -> T = {T:.2f} s, f = {f:.3f} Hz, v = w/k = {w/k:.2f} m/s")

x = np.linspace(0, 12, 1200)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
for phi, ls, lab in [(0.0, "-", r"$\phi = 0$"), (np.pi/2, "--", r"$\phi = \pi/2$"),
                     (np.pi, ":", r"$\phi = \pi$")]:
    ax[0].plot(x, A*np.cos(k*x + phi), ls, lw=1.5, label=lab)
ax[0].annotate("", xy=(0, 1.28), xytext=(lam, 1.28),
               arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.3))
ax[0].text(lam/2, 1.36, r"$\lambda$ = 4 m", ha="center", color="#c0392b", fontsize=9)
ax[0].annotate("", xy=(9.0, 0), xytext=(9.0, 1.0),
               arrowprops=dict(arrowstyle="<->", color="#1e8449", lw=1.3))
ax[0].text(9.15, 0.5, r"$A$ = 1", color="#1e8449", fontsize=9)
ax[0].axhline(0, color="0.4", lw=0.7)
ax[0].set(xlabel="$x$ (m)", ylabel="$y(x,0)$", title=r"(a) snapshot $y(x,0)$ for three phases",
          ylim=(-1.5, 1.6))
ax[0].legend(loc="lower right", ncol=3)

t = np.linspace(0, 6, 1200)
ax[1].plot(t, A*np.cos(-w*t), lw=1.5, color="#1f4e79")
ax[1].annotate("", xy=(0, 1.28), xytext=(T, 1.28),
               arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.3))
ax[1].text(T/2, 1.36, r"$T$ = 2 s  ($f$ = 0.5 Hz)", ha="center", color="#c0392b", fontsize=9)
ax[1].axhline(0, color="0.4", lw=0.7)
ax[1].set(xlabel="$t$ (s)", ylabel="$y(0,t)$", title="(b) time trace $y(0,t)$ at fixed $x=0$",
          ylim=(-1.5, 1.6))
fig.tight_layout()
save(fig, "w3_wave.png")

# vary k and omega independently
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0))
for kk in [k/2, k, 2*k]:
    ax[0].plot(x, np.cos(kk*x), lw=1.3, label=rf"$k$ = {kk:.2f} ($\lambda$ = {2*np.pi/kk:.1f} m)")
ax[0].set(xlabel="$x$ (m)", ylabel="$y(x,0)$",
          title=r"(a) varying $k$ at fixed $t$: spatial compression")
ax[0].legend(fontsize=7.5)
for ww in [w/2, w, 2*w]:
    ax[1].plot(t, np.cos(-ww*t), lw=1.3, label=rf"$\omega$ = {ww:.2f} ($T$ = {2*np.pi/ww:.1f} s)")
ax[1].set(xlabel="$t$ (s)", ylabel="$y(0,t)$",
          title=r"(b) varying $\omega$ at fixed $x$: temporal compression")
ax[1].legend(fontsize=7.5)
fig.tight_layout()
save(fig, "w3_kw.png")

# ======================================================================
# 2. Superposition and interference
# ======================================================================
x = np.linspace(0, 12, 1500)
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
for j, (dphi, ttl) in enumerate([(0.0, r"(a) $\Delta\phi = 0$: constructive"),
                                 (np.pi, r"(b) $\Delta\phi = \pi$: destructive"),
                                 (np.pi/2, r"(c) $\Delta\phi = \pi/2$: partial")]):
    h1, h2 = A*np.cos(k*x), A*np.cos(k*x + dphi)
    ax[j].plot(x, h1, lw=1.0, alpha=0.75, label="$h_1$")
    ax[j].plot(x, h2, lw=1.0, alpha=0.75, ls="--", label="$h_2$")
    ax[j].plot(x, h1+h2, lw=2.0, color="#1e8449", label="$h_1+h_2$")
    ax[j].set(xlabel="$x$ (m)", title=ttl, ylim=(-2.4, 2.4))
    ax[j].axhline(0, color="0.4", lw=0.6)
    amp = 2*A*np.abs(np.cos(dphi/2))
    ax[j].text(0.5, 2.05, f"resultant amp = {amp:.2f}$A$", fontsize=8)
    if j == 0:
        ax[j].set_ylabel("displacement")
    ax[j].legend(fontsize=7, loc="lower right", ncol=3)
fig.tight_layout()
save(fig, "w3_interf.png")

print("\n2  Resultant amplitude = 2A|cos(dphi/2)|:")
for dphi in [0, np.pi/2, np.pi]:
    print(f"     dphi = {dphi:.4f} rad -> amp = {2*np.abs(np.cos(dphi/2)):.4f} A, "
          f"intensity = {(2*np.abs(np.cos(dphi/2)))**2:.4f} A^2")
print("   Energy is redistributed, not destroyed: the spatial average of the")
print("   intensity over a full interference pattern is conserved.")

# N waves with random phases
rng = np.random.default_rng(20044)
print("\n   N random-phase waves: <I> should scale as N (incoherent sum)")
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.7), sharey=False)
for j, N in enumerate([1, 5, 20]):
    ph = rng.uniform(0, 2*np.pi, N)
    tot = sum(A*np.cos(k*x + p) for p in ph)
    I = tot**2
    ax[j].plot(x, I, lw=1.1, color="#8e44ad")
    ax[j].axhline(I.mean(), color="#c0392b", ls="--", lw=1.2,
                  label=f"mean $I$ = {I.mean():.2f}")
    ax[j].axhline(N*A**2/2, color="#1e8449", ls=":", lw=1.4,
                  label=f"$NA^2/2$ = {N*A**2/2:.2f}")
    ax[j].set(xlabel="$x$ (m)", title=f"$N$ = {N}")
    if j == 0:
        ax[j].set_ylabel("intensity $I = h^2$")
    ax[j].legend(fontsize=7)
    print(f"     N = {N:2d}: mean I = {I.mean():6.3f} ; N A^2/2 = {N*A**2/2:6.3f} ; "
          f"peak I = {I.max():6.3f} ; coherent max would be {(N*A)**2:6.1f}")
fig.tight_layout()
save(fig, "w3_random.png")

# statistical convergence
Ns = np.arange(1, 201)
means = []
for N in Ns:
    acc = 0.0
    for _ in range(40):
        ph = rng.uniform(0, 2*np.pi, N)
        acc += (sum(A*np.cos(k*x + p) for p in ph)**2).mean()
    means.append(acc/40)
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.plot(Ns, means, "o", ms=2.6, alpha=0.6, color="#1f4e79", label="ensemble-averaged $\\langle I\\rangle$")
ax.plot(Ns, Ns*A**2/2, "-", lw=1.6, color="#c0392b", label="$N A^2/2$ (incoherent prediction)")
ax.plot(Ns, (Ns*A)**2/2, "--", lw=1.3, color="#1e8449", label="$N^2A^2/2$ (coherent, for contrast)")
ax.set(xlabel="number of waves $N$", ylabel=r"$\langle I \rangle$",
       title="Incoherent ($\\propto N$) vs coherent ($\\propto N^2$) addition", yscale="log")
ax.legend()
fig.tight_layout()
save(fig, "w3_NvsN2.png")
sl = np.polyfit(np.log(Ns), np.log(means), 1)[0]
print(f"   Fitted power law: <I> ~ N^{sl:.3f}  (theory: N^1.000)")

# ======================================================================
# 3. Gaussian wave packet
# ======================================================================
def packet(x, sigma, k0):
    C = (np.pi * sigma**2) ** (-0.25)
    return C * np.exp(-x**2 / (2*sigma**2)) * np.exp(1j*k0*x)

L, Ngrid = 40.0, 8192
xg = np.linspace(-L/2, L/2, Ngrid, endpoint=False)
dx = xg[1] - xg[0]
sigma0, k0 = 1.5, 5.0
psi = packet(xg, sigma0, k0)
P = np.abs(psi)**2
norm = np.sum(P) * dx
print(f"\n3  Gaussian packet: sigma = {sigma0}, k0 = {k0}, grid dx = {dx:.5f}")
print(f"   Normalisation C = (pi sigma^2)^(-1/4) = {(np.pi*sigma0**2)**-0.25:.6f}")
print(f"   Numerical sum(P)*dx = {norm:.10f}   (should be 1)")

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
m = np.abs(xg) < 8
ax[0].plot(xg[m], psi.real[m], lw=1.0, color="#1f4e79", label=r"Re $\Psi(x)$")
ax[0].plot(xg[m], np.abs(psi)[m], lw=1.8, color="#c0392b", label=r"$|\Psi(x)|$")
ax[0].plot(xg[m], -np.abs(psi)[m], lw=1.0, ls="--", color="#c0392b", alpha=0.6)
ax[0].set(xlabel="$x$", ylabel=r"$\Psi$",
          title=rf"(a) wave packet, $\sigma$ = {sigma0}, $k_0$ = {k0}")
ax[0].legend()
ax[1].plot(xg[m], P[m], lw=1.8, color="#1e8449")
ax[1].fill_between(xg[m], P[m], alpha=0.20, color="#1e8449")
ax[1].set(xlabel="$x$", ylabel=r"$P(x)=|\Psi(x)|^2$",
          title=rf"(b) probability density, $\int P\,dx$ = {norm:.6f}")
fig.tight_layout()
save(fig, "w3_packet.png")

# vary sigma
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for s in [0.4, 1.0, 2.5]:
    Pp = np.abs(packet(xg, s, k0))**2
    ax.plot(xg, Pp, lw=1.6, label=rf"$\sigma$ = {s}  (peak {Pp.max():.3f}, $\int P$ = {np.sum(Pp)*dx:.4f})")
ax.set(xlabel="$x$", ylabel="$P(x)$", xlim=(-8, 8),
       title="Localisation vs packet width (all normalised to unit area)")
ax.legend()
fig.tight_layout()
save(fig, "w3_sigma.png")

# ======================================================================
# 4. Heisenberg uncertainty - numerically, via FFT
# ======================================================================
def uncertainties(sigma, k0=5.0):
    psi = packet(xg, sigma, k0)
    P = np.abs(psi)**2
    P /= np.sum(P)*dx
    xm = np.sum(xg*P)*dx
    x2 = np.sum(xg**2*P)*dx
    dxs = np.sqrt(x2 - xm**2)
    # momentum space via FFT
    phi = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2*np.pi)
    kk = np.fft.fftshift(np.fft.fftfreq(Ngrid, d=dx)) * 2*np.pi
    dk = kk[1]-kk[0]
    Pk = np.abs(phi)**2
    Pk /= np.sum(Pk)*dk
    km = np.sum(kk*Pk)*dk
    k2 = np.sum(kk**2*Pk)*dk
    dks = np.sqrt(k2 - km**2)
    return dxs, dks, xg, P, kk, Pk, km

print("\n4  UNCERTAINTY PRINCIPLE (natural units, hbar = 1, so p = k)")
print(f"   {'sigma':>7} {'Dx (num)':>10} {'sigma/sqrt2':>12} {'Dp (num)':>10} "
      f"{'1/(sqrt2 s)':>12} {'Dx*Dp':>9} {'tutorial s*(1/s)':>17}")
sigmas = [0.25, 0.5, 1.0, 1.5, 3.0, 6.0]
prod = []
for s in sigmas:
    dxs, dks, *_ = uncertainties(s)
    prod.append(dxs*dks)
    print(f"   {s:7.2f} {dxs:10.5f} {s/np.sqrt(2):12.5f} {dks:10.5f} "
          f"{1/(np.sqrt(2)*s):12.5f} {dxs*dks:9.5f} {s*(1/s):17.3f}")
print(f"   Mean product = {np.mean(prod):.6f}  ->  Dx*Dp = 1/2 = hbar/2 exactly.")
print("   The Gaussian SATURATES the Heisenberg bound (minimum-uncertainty state).")
print("   The tutorial's Dx ~ sigma, Dp ~ 1/sigma gives the product ~1, correct to")
print("   within a factor of 2 - the factor comes from the precise definition of")
print("   the standard deviation of a Gaussian.")

fig, ax = plt.subplots(2, 2, figsize=(9.0, 5.0))
for j, s in enumerate([0.5, 3.0]):
    dxs, dks, xg_, P_, kk_, Pk_, km_ = uncertainties(s)
    mx = np.abs(xg_) < 12
    ax[0, j].plot(xg_[mx], P_[mx], lw=1.7, color="#1f4e79")
    ax[0, j].fill_between(xg_[mx], P_[mx], alpha=0.2, color="#1f4e79")
    ax[0, j].axvspan(-dxs, dxs, color="#c0392b", alpha=0.13)
    ax[0, j].set(xlabel="$x$", ylabel="$P(x)$",
                 title=rf"$\sigma$ = {s}: $\Delta x$ = {dxs:.3f}")
    mk = np.abs(kk_ - km_) < 12
    ax[1, j].plot(kk_[mk], Pk_[mk], lw=1.7, color="#1e8449")
    ax[1, j].fill_between(kk_[mk], Pk_[mk], alpha=0.2, color="#1e8449")
    ax[1, j].axvspan(km_-dks, km_+dks, color="#c0392b", alpha=0.13)
    ax[1, j].set(xlabel="$p = \\hbar k$", ylabel="$P(p)$",
                 title=rf"$\Delta p$ = {dks:.3f}  $\Rightarrow$ $\Delta x \Delta p$ = {dxs*dks:.4f}")
fig.suptitle("Position and momentum distributions: narrowing one widens the other", y=1.00)
fig.tight_layout()
save(fig, "w3_uncert.png")

ss = np.logspace(-1, 1, 60)
dxa, dka, pra = [], [], []
for s in ss:
    a_, b_, *_ = uncertainties(s)
    dxa.append(a_); dka.append(b_); pra.append(a_*b_)
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.1))
ax[0].loglog(ss, dxa, "o-", ms=3, lw=1.2, label=r"$\Delta x$ (numerical)")
ax[0].loglog(ss, dka, "s-", ms=3, lw=1.2, label=r"$\Delta p$ (numerical)")
ax[0].loglog(ss, ss/np.sqrt(2), "--", lw=1.0, color="0.4", label=r"$\sigma/\sqrt{2}$")
ax[0].loglog(ss, 1/(np.sqrt(2)*ss), ":", lw=1.2, color="0.4", label=r"$1/(\sqrt{2}\sigma)$")
ax[0].set(xlabel=r"$\sigma$", ylabel="uncertainty", title=r"(a) $\Delta x \propto \sigma$, $\Delta p \propto 1/\sigma$")
ax[0].legend(fontsize=7.5)
ax[1].semilogx(ss, pra, "o", ms=3.4, color="#8e44ad", label=r"$\Delta x\,\Delta p$ (numerical)")
ax[1].axhline(0.5, color="#c0392b", ls="--", lw=1.4, label=r"$\hbar/2$ (Heisenberg bound)")
ax[1].set(xlabel=r"$\sigma$", ylabel=r"$\Delta x\,\Delta p$", ylim=(0, 1.1),
          title="(b) the product is independent of $\\sigma$")
ax[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "w3_product.png")

# non-minimum-uncertainty counterexample: top-hat
tophat = (np.abs(xg) < 2.0).astype(float)
tophat /= np.sqrt(np.sum(np.abs(tophat)**2)*dx)
Pt = np.abs(tophat)**2
dxt = np.sqrt(np.sum(xg**2*Pt)*dx)
phit = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(tophat)))*dx/np.sqrt(2*np.pi)
kk = np.fft.fftshift(np.fft.fftfreq(Ngrid, d=dx))*2*np.pi
Pkt = np.abs(phit)**2; Pkt /= np.sum(Pkt)*(kk[1]-kk[0])
dkt = np.sqrt(np.sum(kk**2*Pkt)*(kk[1]-kk[0]))
print(f"\n   Counter-example - square 'top-hat' wave function of half-width 2:")
print(f"     Dx = {dxt:.4f}, Dp = {dkt:.4f}, product = {dxt*dkt:.4f} >> 0.5")
print("     (formally divergent; the sinc tails make Dp large). Only the Gaussian")
print("     achieves equality, which is why it is the natural QM model packet.")
print("=" * 66)
