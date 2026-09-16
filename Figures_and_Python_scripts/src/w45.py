"""Week 4 & 5 supporting computations and figures."""
import numpy as np
from style import plt, save

print("=" * 66)
print("WEEK 4/5  |  SENSOR NUMBERS AND FIGURES")
print("=" * 66)

# ----------------------------------------------------------------------
# 1. Clock fractional accuracy through history
# ----------------------------------------------------------------------
clocks = [
    ("Sundial / water clock", -1500, 1e-2),
    ("Verge escapement",       1300, 1e-2),
    ("Huygens pendulum",       1656, 1e-4),
    ("Harrison H4",            1761, 3e-6),
    ("Shortt free-pendulum",   1921, 1e-8),
    ("Quartz (Marrison)",      1927, 1e-9),
    ("Essen-Parry Cs beam",    1955, 1e-10),
    ("Cs beam (NBS-6)",        1975, 1e-13),
    ("Cs fountain (NIST-F1)",  1999, 1e-15),
    ("Cs fountain (NIST-F2)",  2014, 1e-16),
    ("Sr optical lattice",     2015, 2e-18),
    ("Al+ quantum-logic",      2019, 9.4e-19),
    ("Sr / Yb (state of art)", 2024, 8e-19),
]
cy = np.array([c[1] for c in clocks], float)
ca = np.array([c[2] for c in clocks], float)
sec_per_age = 13.8e9 * 365.25 * 86400
print("\nClock accuracy -> error accumulated over the age of the Universe (13.8 Gyr):")
for n, y, a in clocks[-4:]:
    print(f"  {n:<24} {y}  frac. unc. {a:.1e} -> {a*sec_per_age:.3g} s "
          f"({a*sec_per_age/3600:.3g} hr)")
print(f"\nGPS: 1 ns timing error -> {2.998e8*1e-9*100:.1f} cm position error")
print(f"Relativistic geodesy: dnu/nu = gh/c^2; for h = 1 cm, "
      f"dnu/nu = {9.81*0.01/(2.998e8)**2:.2e}")
print(f"  -> a 1e-18 clock resolves a height change of "
      f"{1e-18*(2.998e8)**2/9.81*100:.2f} cm")

fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.semilogy(cy, ca, "o-", ms=5, lw=1.2, color="#1f4e79")
for n, y, a in clocks:
    off = (7, 5) if y not in (1927, 2014) else (7, -13)
    ax.annotate(n, (y, a), textcoords="offset points", xytext=off, fontsize=7, color="0.3")
ax.axvspan(1955, 2026, color="#c0392b", alpha=0.06)
ax.text(1600, 3e-18, "atomic era", fontsize=8.5, color="#7b241c")
ax.annotate("", xy=(1955, 1.2e-18), xytext=(1930, 1.2e-18),
            arrowprops=dict(arrowstyle="->", color="#7b241c", lw=1.1))
ax.set(xlabel="year", ylabel="fractional frequency uncertainty $\\Delta\\nu/\\nu$",
       title="Nine orders of magnitude in 70 years: clock accuracy through history",
       xlim=(1250, 2075), ylim=(1e-19, 1e-1))
fig.tight_layout()
save(fig, "w4_clocks.png")

# ----------------------------------------------------------------------
# 2. Newton's rings
# ----------------------------------------------------------------------
lam_nm, R = 589e-9, 1.0
r = np.linspace(0, 4e-3, 900)
th = r**2 / (2*R)                      # air-gap thickness
Ipat = np.sin(2*np.pi*th/lam_nm)**2    # reflected intensity, incl. pi phase shift
mmax = 8
print("\nNewton's rings: r_m = sqrt(m lambda R) for DARK rings (reflected light)")
for m in range(1, 6):
    print(f"  m = {m}: r = {np.sqrt(m*lam_nm*R)*1e3:.3f} mm ; "
          f"gap t = {m*lam_nm/2*1e9:.1f} nm")

XX, YY = np.meshgrid(np.linspace(-3.5e-3, 3.5e-3, 460), np.linspace(-3.5e-3, 3.5e-3, 460))
RR = np.sqrt(XX**2 + YY**2)
IMG = np.sin(2*np.pi*(RR**2/(2*R))/lam_nm)**2
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.5))
ax[0].imshow(IMG, cmap="bone", extent=[-3.5, 3.5, -3.5, 3.5], origin="lower")
ax[0].set(xlabel="$x$ (mm)", ylabel="$y$ (mm)",
          title=f"(a) simulated pattern ($\\lambda$ = 589 nm, $R$ = 1 m)")
ax[0].grid(False)
ax[1].plot(r*1e3, Ipat, lw=1.2, color="#1f4e79")
for m in range(1, mmax):
    ax[1].axvline(np.sqrt(m*lam_nm*R)*1e3, color="#c0392b", ls=":", lw=0.8)
ax[1].set(xlabel="radius $r$ (mm)", ylabel="reflected intensity",
          title="(b) radial profile; dashed lines at $r_m=\\sqrt{m\\lambda R}$")
fig.tight_layout()
save(fig, "w4_newton.png")

# ----------------------------------------------------------------------
# 3. Magnetic field magnitudes
# ----------------------------------------------------------------------
fields = [
    ("MRI magnet", 3.0), ("Earth's field", 5e-5), ("Fridge magnet", 5e-3),
    ("Urban noise", 1e-7), ("Human heart (MCG)", 1e-10),
    ("Human brain (MEG)", 1e-13), ("SQUID floor (1 Hz BW)", 1e-15),
    ("NV ensemble floor", 1e-12), ("NV single centre", 1e-9),
]
print("\nMagnetic-field magnitudes and required dynamic range:")
B_squid, B_earth, B_heart, B_brain = 1e-15, 5e-5, 1e-10, 1e-13
print(f"  Earth / SQUID floor  = {B_earth/B_squid:.1e}  ({np.log10(B_earth/B_squid):.1f} decades)")
print(f"  Heart / SQUID floor  = {B_heart/B_squid:.1e}")
print(f"  Brain / SQUID floor  = {B_brain/B_squid:.1e}  -> SNR ~ 100 in 1 Hz bandwidth")
print(f"  Earth / brain signal = {B_earth/B_brain:.1e}  -> shielding + gradiometry essential")

names = [f[0] for f in fields]
vals = np.array([f[1] for f in fields])
order = np.argsort(vals)
fig, ax = plt.subplots(figsize=(6.6, 3.6))
cols = ["#c0392b" if "SQUID" in names[i] or "NV" in names[i] else "#1f4e79" for i in order]
ax.barh(range(len(vals)), vals[order], color=cols, height=0.62, log=True)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels([names[i] for i in order], fontsize=8)
ax.set(xlabel="magnetic field (T, log scale)", xlim=(1e-16, 1e2),
       title="Twelve orders of magnitude: signals vs quantum-sensor noise floors")
for i, j in enumerate(order):
    ax.text(vals[j]*1.6, i, f"{vals[j]:.0e}", va="center", fontsize=7, color="0.3")
fig.tight_layout()
save(fig, "w4_fields.png")

# ----------------------------------------------------------------------
# 4. SQUID V-Phi characteristic
# ----------------------------------------------------------------------
Phi0 = 2.067833848e-15
phi = np.linspace(-2.5, 2.5, 900)
Ic = 2*np.abs(np.cos(np.pi*phi))
V = np.abs(np.sin(np.pi*phi))
A_loop = 1e-4   # 1 cm^2 pickup loop
print(f"\nSQUID: Phi0 = {Phi0:.4e} Wb")
print(f"  Pickup loop A = 1 cm^2 -> one full fringe = Phi0/A = {Phi0/A_loop:.3e} T "
      f"= {Phi0/A_loop*1e12:.1f} pT")
print(f"  Resolving 1e-6 Phi0 -> dB = {1e-6*Phi0/A_loop:.2e} T = "
      f"{1e-6*Phi0/A_loop*1e15:.3f} fT")
print(f"  Earth's field through 1 cm^2 loop = {5e-5*A_loop/Phi0:.3e} flux quanta")

fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.0))
ax[0].plot(phi, Ic, lw=1.7, color="#1f4e79")
ax[0].set(xlabel=r"$\Phi/\Phi_0$", ylabel=r"$I_c(\Phi)/I_0$",
          title=r"(a) critical current: $I_c=2I_0|\cos(\pi\Phi/\Phi_0)|$")
ax[1].plot(phi, V, lw=1.7, color="#c0392b")
ax[1].axvline(0.25, color="#1e8449", ls="--", lw=1.2)
ax[1].plot(0.25, np.abs(np.sin(np.pi*0.25)), "o", ms=6, color="#1e8449")
ax[1].annotate("steepest slope:\nflux-locked operating point", (0.25, 0.707),
               textcoords="offset points", xytext=(30, 26), fontsize=7.5,
               color="#1e8449", ha="left",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1e8449", lw=0.7),
               arrowprops=dict(arrowstyle="->", color="#1e8449", lw=0.9))
ax[1].set_ylim(0, 1.35)
ax[1].set(xlabel=r"$\Phi/\Phi_0$", ylabel="$V$ (arb.)",
          title=r"(b) voltage oscillates with period $\Phi_0$")
fig.tight_layout()
save(fig, "w4_squid.png")

# ----------------------------------------------------------------------
# 5. Atom vs photon interferometry: de Broglie wavelengths
# ----------------------------------------------------------------------
h, kB, amu = 6.62607015e-34, 1.380649e-23, 1.66053907e-27
def ldb(m_amu, T):
    m = m_amu*amu
    v = np.sqrt(3*kB*T/m)
    return h/(m*v), v
print("\nde Broglie wavelength of Rb-87 vs optical photon (633 nm = 6.33e-7 m):")
for T, lab in [(300, "room temperature"), (1e-4, "laser-cooled, 100 uK"),
               (1e-7, "sub-recoil, 100 nK")]:
    l, v = ldb(87, T)
    print(f"  T = {T:>8.0e} K ({lab:<22}): v = {v:8.3f} m/s, "
          f"lambda_dB = {l:.3e} m = {l*1e9:.4f} nm")
fC = 87*amu*(2.998e8)**2/h
print(f"  Compton frequency of Rb-87: mc^2/h = {fC:.3e} Hz "
      f"({fC/9.19e9:.2e}x the Cs clock frequency)")
keff = 2*2*np.pi/780e-9
for Tint in [0.01, 0.1, 1.0]:
    print(f"  Mach-Zehnder phase for g = 9.81 m/s^2, T = {Tint:5.2f} s: "
          f"dphi = k_eff g T^2 = {keff*9.81*Tint**2:.3e} rad "
          f"({keff*9.81*Tint**2/(2*np.pi):.3e} fringes)")
print(f"  -> resolving 1 mrad of phase at T = 1 s gives dg/g ~ "
        f"{1e-3/(keff*9.81*1.0):.2e}")

# ----------------------------------------------------------------------
# 6. Standard quantum limit vs Heisenberg limit (clocks & squeezing)
# ----------------------------------------------------------------------
Nat = np.logspace(0, 6, 200)
fig, ax = plt.subplots(figsize=(5.8, 3.1))
ax.loglog(Nat, 1/np.sqrt(Nat), lw=1.7, color="#1f4e79",
          label=r"standard quantum limit $\propto 1/\sqrt{N}$")
ax.loglog(Nat, 1/Nat, lw=1.7, color="#c0392b",
          label=r"Heisenberg limit $\propto 1/N$ (entangled)")
ax.fill_between(Nat, 1/Nat, 1/np.sqrt(Nat), color="#1e8449", alpha=0.12)
ax.text(3e2, 4e-3, "accessible only with\nentanglement / squeezing",
        fontsize=8, color="#1e8449")
ax.set(xlabel="number of atoms $N$", ylabel="fractional frequency instability (arb.)",
       title="Why entanglement matters for sensing")
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "w4_sql.png")
for N_ in [1e2, 1e4, 1e6]:
    print(f"  N = {N_:.0e}: SQL = {1/np.sqrt(N_):.2e}, Heisenberg = {1/N_:.2e}, "
          f"gain = {np.sqrt(N_):.0f}x")

# ----------------------------------------------------------------------
# 7. Week 5: LIGO strain sensitivity
# ----------------------------------------------------------------------
L_arm, h_strain = 4000.0, 1e-21
print(f"\nLIGO: arm length {L_arm:.0f} m, strain h ~ {h_strain:.0e}")
print(f"  -> displacement dL = h*L = {h_strain*L_arm:.2e} m "
      f"= {h_strain*L_arm/1e-15:.2e} fm = {h_strain*L_arm/0.84e-15:.4f} proton radii")
print(f"  Squeezed-light upgrade (2019): ~3 dB -> range x{10**(3/20):.2f}, "
      f"detection rate x{(10**(3/20))**3:.2f}")
f_ = np.logspace(1, 3.5, 400)
shot = 1e-23*np.sqrt(1+(f_/100)**2)
rad = 3e-21*(30/f_)**2
seis = 1e-19*(20/f_)**6
tot = np.sqrt(shot**2+rad**2+seis**2)
sq = np.sqrt((shot/10**(3/20))**2+(rad*10**(3/20))**2+seis**2)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.loglog(f_, tot, lw=1.8, color="#1f4e79", label="total (coherent light)")
ax.loglog(f_, sq, lw=1.8, color="#c0392b", label="with squeezed vacuum injection")
ax.loglog(f_, shot, ":", lw=1.1, color="0.45", label="photon shot noise")
ax.loglog(f_, rad, "--", lw=1.1, color="0.45", label="radiation-pressure noise")
ax.loglog(f_, seis, "-.", lw=1.1, color="0.45", label="seismic (schematic)")
ax.set(xlabel="frequency (Hz)", ylabel=r"strain noise (Hz$^{-1/2}$)",
       title="Quantum noise in a gravitational-wave detector (schematic)",
       ylim=(1e-24, 1e-19))
ax.legend(fontsize=7.5)
fig.tight_layout()
save(fig, "w5_ligo.png")
print("=" * 66)
