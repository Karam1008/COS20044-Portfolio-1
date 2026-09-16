"""Week 2 artefact: numerical verification of the pen-and-paper linear algebra."""
import numpy as np
from style import plt, save
np.set_printoptions(precision=4, suppress=True)

print("=" * 66)
print("WEEK 2  |  VERIFICATION OF HAND CALCULATIONS")
print("=" * 66)

# 1. Complex numbers
z1, z2 = 2 + 3j, 1 - 1j
print(f"1(a) z1*z2            = {z1*z2}          [hand: 5 + i]")
z1, z2 = 3 + 4j, 1 - 2j
print(f"1(b) z1+z2            = {z1+z2}          [hand: 4 + 2i]")
print(f"     z1-z2            = {z1-z2}          [hand: 2 + 6i]")
z1, z2 = 1 + 2j, 3 - 1j
print(f"1(c) z1/z2            = {z1/z2}     [hand: 0.1 + 0.7i]")
print(f"     |z1|             = {abs(z1):.6f}       [hand: sqrt(5) = {np.sqrt(5):.6f}]")

# 2. Vectors
a, b = np.array([3, -1]), np.array([-2, 4])
print(f"\n2(a) a+b              = {a+b}            [hand: (1, 3)]")
print(f"     3a-2b            = {3*a-2*b}         [hand: (13, -11)]")
v = np.array([5, -12])
print(f"2(b) |v|              = {np.linalg.norm(v):.4f}           [hand: 13]")
print(f"     v_hat            = {v/np.linalg.norm(v)}   [hand: (5/13, -12/13)]")
u, w = np.array([2, 3]), np.array([4, -1])
print(f"2(c) u.w              = {u@w}                [hand: 5]")

# 3. Matrices
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[0, 1], [1, 0]], dtype=complex)      # NB: this is Pauli-X

print("\n3   HY  =\n", H @ Y, "\n     [hand: (i/sqrt2) [[1,-1],[-1,-1]] ]")
print("    check vs (i/sqrt2)[[1,-1],[-1,-1]]:",
      np.allclose(H @ Y, 1j/np.sqrt(2)*np.array([[1, -1], [-1, -1]])))
print("    ZH  =\n", Z @ H, "\n     [hand: (1/sqrt2) [[1,1],[-1,1]] ]")
print("    check:", np.allclose(Z @ H, 1/np.sqrt(2)*np.array([[1, 1], [-1, 1]])))
print("    S^2 =\n", S @ S, "  [hand: identity]")
print("    HY == YH ?", np.allclose(H@Y, Y@H), "-> gates do NOT commute")
print("    commutator [H,Y] =\n", H@Y - Y@H)

# unitarity / hermiticity
for nm, M in [("H", H), ("Y", Y), ("Z", Z), ("S", S)]:
    print(f"    {nm}: Hermitian={np.allclose(M, M.conj().T)}  "
          f"Unitary={np.allclose(M@M.conj().T, np.eye(2))}  "
          f"M^2=I: {np.allclose(M@M, np.eye(2))}  det={np.linalg.det(M).real:+.3f}  "
          f"tr={np.trace(M).real:+.3f}")

# 4. Eigenproblems
print("\n4   EIGENVALUES / EIGENVECTORS")
for nm, M in [("H", H), ("Y", Y), ("Z", Z), ("S(=X)", S)]:
    ev, evec = np.linalg.eigh(M)
    print(f"  {nm}: eigenvalues {ev.round(6)}")
    for i in range(2):
        v_ = evec[:, i]
        # fix global phase so first non-zero entry is real positive
        idx = np.argmax(np.abs(v_) > 1e-9)
        v_ = v_ * np.exp(-1j*np.angle(v_[idx]))
        print(f"      lambda = {ev[i]:+.4f} -> {np.round(v_, 4)}")

s2 = np.sqrt(2)
print(f"\n  H eigenvector check (lambda=+1): (1, sqrt2-1)/norm = "
      f"{np.array([1, s2-1])/np.linalg.norm([1, s2-1])}")
print(f"  H eigenvector check (lambda=-1): (1, -(sqrt2+1))/norm = "
      f"{np.array([1, -(s2+1)])/np.linalg.norm([1, -(s2+1)])}")
print(f"  Bloch-axis angle of H: theta = {np.degrees(np.arctan2(1,1)/1):.1f} deg from z "
      f"(axis (x+z)/sqrt2), half-angle {np.degrees(np.arctan(s2-1)):.2f} deg")

# --- Extension problems (asked of the genAI, solved by hand, checked here) ---
print("\n--- EXTENSION PROBLEMS ----------------------------------------")
X, I2 = S, np.eye(2)
n = np.array([1, 1, 1]) / np.sqrt(3)
sigma = [X, Y, Z]
nsig = sum(n[i]*sigma[i] for i in range(3))
ev, evec = np.linalg.eigh(nsig)
print(f"E1  n.sigma with n=(1,1,1)/sqrt3 : eigenvalues {ev.round(6)}  (expect +-1)")
theta = np.pi/3
U = np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*nsig
print(f"E2  U = exp(-i(theta/2) n.sigma), theta=60deg : unitary? "
      f"{np.allclose(U@U.conj().T, I2)}  det = {np.linalg.det(U):.4f}")
psi = np.array([1, 0], dtype=complex)
out = H @ psi
print(f"E3  H|0> = {out.round(4)}  -> P(0) = {abs(out[0])**2:.3f}, P(1) = {abs(out[1])**2:.3f}")
bell = np.kron(H, I2) @ np.array([1, 0, 0, 0], dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
bell = CNOT @ bell
print(f"E4  CNOT(H x I)|00> = {bell.round(4)}  -> Bell state (|00>+|11>)/sqrt2")
rho = np.outer(bell, bell.conj())
rhoA = rho.reshape(2,2,2,2).trace(axis1=1, axis2=3)
print(f"    reduced rho_A = \n{rhoA.round(4)}  ;  purity Tr(rho_A^2) = "
      f"{np.trace(rhoA@rhoA).real:.4f}  (0.5 = maximally entangled)")
print(f"E5  Tr(HY) = {np.trace(H@Y):.4f} ; Tr(YH) = {np.trace(Y@H):.4f} (cyclic property holds)")

# --- Figure: Bloch sphere with the eigenvectors ------------------------
fig = plt.figure(figsize=(4.6, 4.4))
ax = fig.add_subplot(111, projection="3d")
u_, v_ = np.mgrid[0:2*np.pi:80j, 0:np.pi:40j]
ax.plot_surface(np.cos(u_)*np.sin(v_), np.sin(u_)*np.sin(v_), np.cos(v_),
                color="#8ea9c1", alpha=0.13, linewidth=0)
for ang in np.linspace(0, np.pi, 7):
    ax.plot(np.cos(np.linspace(0, 2*np.pi, 100))*np.sin(ang),
            np.sin(np.linspace(0, 2*np.pi, 100))*np.sin(ang),
            np.full(100, np.cos(ang)), color="0.7", lw=0.35)
axes_ = {"x": ([1,0,0], "#c0392b"), "y": ([0,1,0], "#1e8449"), "z": ([0,0,1], "#1f4e79")}
for lab, (vec, col) in axes_.items():
    ax.quiver(0, 0, 0, *vec, color=col, lw=1.5, arrow_length_ratio=0.12)
    ax.text(vec[0]*1.22, vec[1]*1.22, vec[2]*1.22, lab, color=col, fontsize=11)
ax.text(0, 0, 1.12, r"$|0\rangle$", fontsize=9, color="#1f4e79")
ax.text(0, 0, -1.30, r"$|1\rangle$", fontsize=9, color="#1f4e79")
ax.text(1.20, 0, -0.12, r"$|+\rangle$", fontsize=9, color="#c0392b")
ax.text(0, 1.20, -0.12, r"$|{+}i\rangle$", fontsize=9, color="#1e8449")
h_ax = np.array([1, 0, 1])/np.sqrt(2)
ax.quiver(0, 0, 0, *h_ax, color="#8e44ad", lw=2.4, arrow_length_ratio=0.14)
ax.quiver(0, 0, 0, *(-h_ax), color="#8e44ad", lw=2.4, arrow_length_ratio=0.14, alpha=0.6)
ax.text(h_ax[0]*1.25, 0, h_ax[2]*1.25, "H axis\n(22.5$^\\circ$)", fontsize=8, color="#8e44ad")
ax.set_box_aspect([1,1,1]); ax.set_axis_off()
ax.view_init(elev=18, azim=32)
fig.tight_layout()
save(fig, "w2_bloch.png")
print("=" * 66)
