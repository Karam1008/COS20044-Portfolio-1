"""Week 1 artefact: Moore's Law - numerical investigation."""
import numpy as np
from style import plt, save

# ----------------------------------------------------------------------
# 1. Microprocessor transistor-count dataset
#    (representative flagship parts; counts as published by the vendors)
# ----------------------------------------------------------------------
data = [
    ("Intel 4004",             1971, 2.30e3),
    ("Intel 8008",             1972, 3.50e3),
    ("Intel 8080",             1974, 4.50e3),
    ("Intel 8086",             1978, 2.90e4),
    ("Intel 80286",            1982, 1.34e5),
    ("Intel 80386",            1985, 2.75e5),
    ("Intel 80486",            1989, 1.18e6),
    ("Pentium",                1993, 3.10e6),
    ("Pentium Pro",            1995, 5.50e6),
    ("Pentium II",             1997, 7.50e6),
    ("Pentium III",            1999, 9.50e6),
    ("Pentium 4 (Willamette)", 2000, 4.20e7),
    ("Itanium 2 (Madison)",    2003, 4.10e8),
    ("Core 2 Duo (Conroe)",    2006, 2.91e8),
    ("Core i7 (Nehalem)",      2008, 7.31e8),
    ("Core i7 (Gulftown)",     2010, 1.17e9),
    ("Xeon (Ivy Bridge-EX)",   2014, 4.31e9),
    ("Xeon (Broadwell-EP)",    2016, 7.20e9),
    ("Apple A12X",             2018, 1.00e10),
    ("AMD EPYC Rome",          2019, 3.95e10),
    ("Apple M1 Ultra",         2022, 1.14e11),
    ("AMD MI300X",             2023, 1.53e11),
    ("NVIDIA B200",            2024, 2.08e11),
]
name = [d[0] for d in data]
year = np.array([d[1] for d in data], float)
N = np.array([d[2] for d in data], float)

# ----------------------------------------------------------------------
# 2. Least-squares fit of log2(N) = a*(t - t0) + b  ->  T_double = 1/a
# ----------------------------------------------------------------------
def fit(yr, n):
    a, b = np.polyfit(yr - 1971.0, np.log2(n), 1)
    resid = np.log2(n) - (a * (yr - 1971.0) + b)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((np.log2(n) - np.log2(n).mean()) ** 2)
    return a, b, 1.0 / a, 1 - ss_res / ss_tot, resid

a_all, b_all, Td_all, R2_all, res_all = fit(year, N)

m_early = year <= 2002
m_late = year >= 2003
a_e, b_e, Td_e, R2_e, _ = fit(year[m_early], N[m_early])
a_l, b_l, Td_l, R2_l, _ = fit(year[m_late], N[m_late])

print("=" * 66)
print("WEEK 1  |  MOORE'S LAW FIT")
print("=" * 66)
print(f"Full range 1971-2024 : doubling time = {Td_all:6.3f} yr   R^2 = {R2_all:.4f}")
print(f"  N(t) = {2**b_all:.3g} x 2^[(t-1971)/{Td_all:.3f}]")
print(f"Early 1971-2002      : doubling time = {Td_e:6.3f} yr   R^2 = {R2_e:.4f}")
print(f"Late  2003-2024      : doubling time = {Td_l:6.3f} yr   R^2 = {R2_l:.4f}")
print(f"Max residual         : {np.max(np.abs(res_all)):.2f} doublings "
      f"(factor {2**np.max(np.abs(res_all)):.1f}) at {name[int(np.argmax(np.abs(res_all)))]}")
print(f"Total growth 1971-2024: {N[-1]/N[0]:.3e}x over {year[-1]-year[0]:.0f} yr "
      f"= {np.log2(N[-1]/N[0]):.1f} doublings")

# --- Figure 1.1  exponential growth on linear vs log axes -------------
t = np.linspace(0, 30, 400)
y = 2.0 ** (t / 2.0)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax[0].plot(t, y, lw=1.8)
ax[0].set(xlabel="time (years)", ylabel="$N(t)=2^{t/2}$", title="(a) linear axes")
ax[1].semilogy(t, y, lw=1.8)
ax[1].set(xlabel="time (years)", ylabel="$N(t)$ (log scale)", title="(b) semi-log axes")
for k in (0, 1):
    ax[k].set_xlim(0, 30)
fig.tight_layout()
save(fig, "w1_exp.png")

# --- Figure 1.2  Moore's law with fit ---------------------------------
tt = np.linspace(1969, 2042, 300)
fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.semilogy(year, N, "o", ms=5.5, color="#1f4e79", label="measured transistor count")
ax.semilogy(tt, 2 ** (a_all * (tt - 1971) + b_all), "-", lw=1.5, color="#c0392b",
            label=f"global fit: $T_2$ = {Td_all:.2f} yr ($R^2$ = {R2_all:.3f})")
ax.semilogy(tt[tt <= 2010], 2 ** (a_e * (tt[tt <= 2010] - 1971) + b_e), "--", lw=1.2,
            color="#1e8449", label=f"1971--2002 fit: $T_2$ = {Td_e:.2f} yr")
ax.semilogy(tt[tt >= 2000], 2 ** (a_l * (tt[tt >= 2000] - 1971) + b_l), "--", lw=1.2,
            color="#8e44ad", label=f"2003--2024 fit: $T_2$ = {Td_l:.2f} yr")
ax.axvspan(2024, 2042, color="0.85", alpha=0.5, zorder=0)
ax.text(2033, 3e4, "extrapolation", ha="center", fontsize=8.5, color="0.35")
for lbl, x, yv in [("4004", 1971, 2.3e3), ("8086", 1978, 2.9e4), ("486", 1989, 1.18e6),
                   ("Pentium 4", 2000, 4.2e7), ("Core 2", 2006, 2.91e8),
                   ("M1 Ultra", 2022, 1.14e11)]:
    ax.annotate(lbl, (x, yv), textcoords="offset points", xytext=(6, -11), fontsize=7.5,
                color="0.30")
ax.set(xlabel="year of introduction", ylabel="transistors per chip",
       title="Moore's law, 1971--2024, with exponential fits and extrapolation to 2040")
ax.set_xlim(1969, 2042)
ax.set_ylim(1e3, 1e14)
ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "w1_moore.png")

# --- Figure 1.3  residuals --------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.bar(year, res_all, width=1.1, color=["#1f4e79" if r >= 0 else "#c0392b" for r in res_all])
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="year", ylabel="residual (doublings)",
       title="Residuals of the global fit: $\\log_2 N_{\\rm measured}-\\log_2 N_{\\rm fit}$")
fig.tight_layout()
save(fig, "w1_resid.png")

# ----------------------------------------------------------------------
# 3. Extrapolation to 2040 and the atomic limit
# ----------------------------------------------------------------------
N2040 = 2 ** (a_all * (2040 - 1971) + b_all)
A_die = 8.0e-4                       # m^2, a generous 800 mm^2 reticle-limited die
a_Si = 5.431e-10                     # m, silicon lattice constant
n_surface = (1.0 / a_Si) ** 2        # surface atoms per m^2 (one atomic layer)
atoms_die = n_surface * A_die
area_per_tr_2024 = A_die / N[-1]
area_per_tr_2040 = A_die / N2040
print("\n--- Extrapolation to 2040 -------------------------------------")
print(f"Fit predicts N(2040) = {N2040:.3e} transistors on one die")
print(f"Area/transistor 2024 = {area_per_tr_2024:.3e} m^2 "
      f"-> square of side {np.sqrt(area_per_tr_2024)*1e9:.1f} nm")
print(f"Area/transistor 2040 = {area_per_tr_2040:.3e} m^2 "
      f"-> square of side {np.sqrt(area_per_tr_2040)*1e9:.3f} nm")
print(f"Si lattice constant  = {a_Si*1e9:.3f} nm; Si-Si bond = 0.235 nm")
print(f"Surface Si atoms on an 800 mm^2 die (1 layer) = {atoms_die:.3e}")
print(f"Transistors would outnumber surface atoms by {N2040/atoms_die:.1f}x")
print(f"Atoms per transistor in 2040 = {atoms_die/N2040:.3f}  (<1 => impossible)")
N_atomlimit_year = (np.log2(atoms_die) - b_all) / a_all + 1971
print(f"Fit crosses the 1-atom-per-transistor line in year {N_atomlimit_year:.1f}")

# --- Figure 1.4  atomic limit ------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.5))
tt2 = np.linspace(1971, 2060, 400)
ax.semilogy(tt2, 2 ** (a_all * (tt2 - 1971) + b_all), color="#c0392b", lw=1.5,
            label="Moore's law extrapolated")
ax.semilogy(year, N, "o", ms=4.5, color="#1f4e79", label="real devices")
ax.axhline(atoms_die, color="#1e8449", ls="--", lw=1.4,
           label=f"1 transistor per surface Si atom ({atoms_die:.1e})")
ax.axvline(N_atomlimit_year, color="0.4", ls=":", lw=1.2)
ax.text(N_atomlimit_year + 0.8, 1e5, f"hard atomic wall\n$\\approx$ {N_atomlimit_year:.0f}",
        fontsize=8.5, color="0.3")
ax.fill_between(tt2, atoms_die, 1e18, color="#c0392b", alpha=0.07)
ax.text(1990, 1e16, "physically forbidden region\n(sub-atomic transistors)",
        fontsize=8.5, color="#7b241c")
ax.set(xlabel="year", ylabel="transistors per 800 mm$^2$ die",
       title="Why Moore's law cannot reach 2040 in its original form")
ax.set_ylim(1e3, 1e18)
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "w1_atomic.png")

# ----------------------------------------------------------------------
# 4. Banana flies
# ----------------------------------------------------------------------
weeks = (2026.0 - 1965.0) * 365.2425 / 7.0
n_flies_log10 = weeks * np.log10(2.0)
m_fly = 1.0e-6                        # kg, ~1 mg for Drosophila melanogaster
mass_log10 = n_flies_log10 + np.log10(m_fly)
M_earth, M_sun, M_universe = 5.972e24, 1.989e30, 1.5e53
print("\n--- Banana flies ----------------------------------------------")
print(f"Elapsed weeks 1965 -> 2026 : {weeks:.1f}")
print(f"Number of flies  = 2^{weeks:.0f} = 10^{n_flies_log10:.1f}")
print(f"Biomass (1 mg each) = 10^{mass_log10:.1f} kg")
print(f"  / Earth mass       = 10^{mass_log10-np.log10(M_earth):.1f}")
print(f"  / Sun mass         = 10^{mass_log10-np.log10(M_sun):.1f}")
print(f"  / observable Univ. = 10^{mass_log10-np.log10(M_universe):.1f}")
t_earth = np.log2(M_earth / m_fly) * 7 / 365.2425
print(f"Time to reach Earth's mass = {np.log2(M_earth/m_fly):.0f} weeks = {t_earth:.1f} yr "
      f"(i.e. by {1965+t_earth:.0f})")

# --- Figure 1.5  fly biomass -------------------------------------------
yrs = np.linspace(0, 61, 500)
logm = yrs * 365.2425 / 7 * np.log10(2) + np.log10(m_fly)
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.plot(1965 + yrs, logm, lw=1.8, color="#d68910")
for lab, M, col in [("Earth", M_earth, "#1e8449"), ("Sun", M_sun, "#c0392b"),
                    ("Milky Way", 1.5e42, "#8e44ad"),
                    ("observable Universe", M_universe, "#1f4e79")]:
    ax.axhline(np.log10(M), ls="--", lw=1.0, color=col)
    ax.text(1966, np.log10(M) + 12, lab, fontsize=8, color=col)
ax.set(xlabel="year", ylabel=r"$\log_{10}$ (biomass / kg)",
       title="Unbounded weekly doubling: fruit-fly biomass, 1965--2026")
ax.set_xlim(1965, 2026)
fig.tight_layout()
save(fig, "w1_flies.png")

# ----------------------------------------------------------------------
# 5. Dennard scaling / power density / clock plateau
# ----------------------------------------------------------------------
cpu = [
    # name, year, clock GHz, TDP W, die mm^2, Vdd
    ("i486DX",      1989, 0.025,   3, 163, 5.0),
    ("Pentium",     1993, 0.066,  16, 294, 5.0),
    ("Pentium II",  1997, 0.300,  43, 203, 2.8),
    ("Pentium III", 1999, 0.600,  34, 128, 1.8),
    ("Pentium 4",   2000, 1.500,  55, 217, 1.75),
    ("P4 Prescott", 2004, 3.800, 115, 112, 1.4),
    ("Core 2 Duo",  2006, 2.930,  65, 143, 1.35),
    ("Core i7-980X",2010, 3.330, 130, 248, 1.35),
    ("Core i7-4770K",2013,3.500,  84, 177, 1.20),
    ("Core i9-9900K",2018,3.600,  95, 174, 1.20),
    ("Core i9-13900K",2022,3.000,125, 257, 1.15),
]
cy = np.array([c[1] for c in cpu], float)
ck = np.array([c[2] for c in cpu], float)
tdp = np.array([c[3] for c in cpu], float)
die = np.array([c[4] for c in cpu], float)
vdd = np.array([c[5] for c in cpu], float)
pdens = tdp / (die * 1e-2)            # W/cm^2

print("\n--- Power / clock ---------------------------------------------")
for c, p in zip(cpu, pdens):
    print(f"  {c[0]:<16} {c[1]}  f={c[2]:5.3f} GHz  Vdd={c[5]:.2f} V  "
          f"P/A={p:6.1f} W/cm^2")
print(f"Clock growth 1989-2004: {ck[5]/ck[0]:.0f}x in 15 yr "
      f"({np.log2(ck[5]/ck[0])/15*12:.1f} doublings/decade)")
print(f"Clock growth 2004-2022: {ck[-1]/ck[5]:.2f}x in 18 yr  -> plateau")
print(f"Hotplate ~ 10 W/cm^2; nuclear reactor core ~ 100 W/cm^2")

fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
ax[0].semilogy(cy, ck, "o-", ms=4.5, lw=1.3)
ax[0].axvline(2005, color="0.5", ls=":", lw=1.2)
ax[0].text(2005.7, 0.05, "clock plateau", fontsize=8, color="0.35", rotation=90)
ax[0].set(xlabel="year", ylabel="clock frequency (GHz)", title="(a) clock speed")
ax[1].plot(cy, vdd, "s-", ms=4.5, lw=1.3, color="#c0392b")
ax[1].axhline(0.7, color="0.5", ls="--", lw=1.0)
ax[1].text(1992, 0.78, r"$\sim$ threshold-voltage floor", fontsize=7.5, color="0.35")
ax[1].set(xlabel="year", ylabel=r"supply voltage $V_{dd}$ (V)", title="(b) voltage scaling stalls")
ax[1].set_ylim(0, 5.6)
ax[2].plot(cy, pdens, "^-", ms=4.5, lw=1.3, color="#1e8449")
ax[2].axhline(10, color="0.5", ls="--", lw=1.0)
ax[2].text(1990, 12, "hotplate ($\\sim$10 W/cm$^2$)", fontsize=7.5, color="0.35")
ax[2].set(xlabel="year", ylabel="power density (W/cm$^2$)", title="(c) thermal wall")
fig.tight_layout()
save(fig, "w1_dennard.png")

# --- Figure 1.7  gate-oxide tunnelling ---------------------------------
hbar, me, q = 1.054571817e-34, 9.1093837e-31, 1.602176634e-19
phi = 3.1 * q            # Si/SiO2 conduction-band offset, J
meff = 0.5 * me          # electron effective mass in SiO2
kappa = np.sqrt(2 * meff * phi) / hbar
d = np.linspace(0.5e-9, 5e-9, 400)
T = np.exp(-2 * kappa * d)
print("\n--- Gate-oxide tunnelling -------------------------------------")
print(f"kappa = {kappa:.3e} 1/m ; decay length 1/(2 kappa) = {1/(2*kappa)*1e9:.3f} nm")
for dd in (3e-9, 2e-9, 1.5e-9, 1.2e-9, 1.0e-9):
    print(f"  d = {dd*1e9:.1f} nm -> T = {np.exp(-2*kappa*dd):.3e}  "
          f"(x{np.exp(-2*kappa*dd)/np.exp(-2*kappa*3e-9):.3g} vs 3 nm)")

fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.semilogy(d * 1e9, T, lw=1.8, color="#8e44ad")
for dd, lab in [(3.0, "1997 node"), (1.2, "2003 node")]:
    ax.plot(dd, np.exp(-2 * kappa * dd * 1e-9), "o", ms=6, color="#c0392b")
    ax.annotate(f"{lab}\n$d$ = {dd} nm", (dd, np.exp(-2 * kappa * dd * 1e-9)),
                textcoords="offset points", xytext=(10, 4), fontsize=8)
ax.set(xlabel="gate-oxide thickness $d$ (nm)",
       ylabel=r"tunnelling transmission $T \sim e^{-2\kappa d}$",
       title=r"Direct tunnelling through the gate oxide ($\phi_B$ = 3.1 eV, $m^*$ = 0.5$m_e$)")
fig.tight_layout()
save(fig, "w1_tunnel.png")

# ----------------------------------------------------------------------
# 6. genAI scaling
# ----------------------------------------------------------------------
models = [
    ("Perceptron",   1958, 6.9e5),
    ("NetTalk",      1987, 8.1e9),
    ("LeNet-5",      1998, 3.8e13),
    ("AlexNet",      2012, 4.7e17),
    ("Seq2Seq",      2014, 4.0e18),
    ("ResNet-152",   2015, 1.1e19),
    ("GPT-1",        2018, 1.8e19),
    ("BERT-Large",   2018, 2.6e20),
    ("GPT-2",        2019, 1.5e21),
    ("GPT-3",        2020, 3.1e23),
    ("Gopher",       2021, 6.3e23),
    ("Chinchilla",   2022, 5.8e23),
    ("PaLM",         2022, 2.5e24),
    ("GPT-4 (est.)", 2023, 2.1e25),
    ("Frontier 2024",2024, 5.0e25),
]
my = np.array([m[1] for m in models], float)
mf = np.array([m[2] for m in models], float)
pre, post = my < 2010, my >= 2010
ap, bp = np.polyfit(my[pre], np.log2(mf[pre]), 1)
aq, bq = np.polyfit(my[post], np.log2(mf[post]), 1)
print("\n--- Training-compute scaling ----------------------------------")
print(f"Pre-2010  doubling time = {1/ap*12:.1f} months")
print(f"Post-2010 doubling time = {1/aq*12:.1f} months")
print(f"Ratio to Moore's law ({Td_all*12:.0f} months): {(Td_all*12)/(1/aq*12):.1f}x faster")

fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.semilogy(my[pre], mf[pre], "o", ms=5, color="#1f4e79", label="pre-2010 era")
ax.semilogy(my[post], mf[post], "s", ms=5, color="#c0392b", label="deep-learning era")
xa = np.linspace(1955, 2012, 50); xb = np.linspace(2010, 2032, 50)
ax.semilogy(xa, 2 ** (ap * xa + bp), "-", lw=1.2, color="#1f4e79",
            label=f"fit: $T_2$ = {1/ap*12:.0f} months")
ax.semilogy(xb, 2 ** (aq * xb + bq), "-", lw=1.2, color="#c0392b",
            label=f"fit: $T_2$ = {1/aq*12:.1f} months")
ax.semilogy(xb, 2 ** (a_all * (xb - 1971) + b_all) / N[0] * mf[post][0] * 1e-3, ":",
            lw=1.3, color="#1e8449", label="Moore's-law slope for comparison")
for lab, x, yv in [("AlexNet", 2012, 4.7e17), ("GPT-3", 2020, 3.1e23),
                   ("GPT-4 (est.)", 2023, 2.1e25)]:
    ax.annotate(lab, (x, yv), textcoords="offset points", xytext=(-46, 6), fontsize=7.5)
ax.set(xlabel="year", ylabel="training compute (FLOP)",
       title="Training compute of notable ML systems vs Moore's-law slope")
ax.set_xlim(1955, 2032)
ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "w1_ai.png")

# data-centre energy
yrs_e = np.array([2015, 2018, 2020, 2022, 2024, 2026])
twh = np.array([200, 250, 300, 460, 620, 1000.0])
fig, ax = plt.subplots(figsize=(6.0, 2.9))
ax.bar(yrs_e[:-2], twh[:-2], width=1.3, color="#1f4e79", label="reported")
ax.bar(yrs_e[-2:], twh[-2:], width=1.3, color="#c0392b", alpha=0.75, label="projected")
ax.set(xlabel="year", ylabel="electricity use (TWh yr$^{-1}$)",
       title="Global data-centre electricity demand (order-of-magnitude)")
ax.legend()
fig.tight_layout()
save(fig, "w1_energy.png")
print("\nAustralia's total electricity consumption is ~265 TWh/yr for scale.")
print("=" * 66)
