# Symbolic Regression

Equation discovery for **Chapter 10 (Machine Learning in Computational Mechanics)**:
closed-form laws recovered from sampled data, each with a case where the method fails.
SINDy and EUCLID share the sequential thresholded least squares in `helper.py`.

## Drivers

- `ai_feynman.py`
  dimensional analysis plus brute-force search over power laws (simplified AI Feynman),
  rediscovering the Euler buckling load and the first cantilever eigenfrequency, and
  failing when the density is left out
- `sindy.py`
  sparse identification of the Lorenz system, the identified and true trajectories,
  identification error versus noise and versus the length of the measured window, and a
  pendulum identified with and without $\sin q$ in the library
- `euclid.py`
  unsupervised identification of a nonlinear elastic law from displacements and reaction
  forces of a tapered bar, failing for a prismatic bar and for a too large library

## Non-obvious technicalities (authored by Claude)

**Dimensionless groups.** The dimension matrix $\mathbf{D}$ holds the exponents of mass,
length, and time of every variable in its columns. Its rank, from the singular values,
sets the number of dimensionless groups (Buckingham $\Pi$ theorem): variables minus rank.
A nullspace basis from the SVD is correct but unreadable (irrational mixtures), so the
groups are built with repeating variables instead: the first inputs with independent
units form a basis, the target and every remaining input are divided by the product of
basis powers that carries the same units. If the target units are not in the span of the
input units (least-squares residual > 0), no group contains the target, which is how
dropping the density from the cantilever is detected before any fit.

**Brute force.** In log space every candidate power law is linear, so the best constant
of each exponent combination is the mean log residual and its relative error the standard
deviation. The search over $9^G$ combinations of $G$ groups replaces $9^{n}$ over the $n$
raw inputs (81 instead of 59049 for the cantilever). A constant is accepted as known if
it lies within 1 % of a short list; 0.5596 $= 1.8751^2/(2\pi)$ comes from the root of
$\cos\beta\cosh\beta=-1$ and stays numerical. A tolerance that is too loose would
misname it, e.g. $1/\sqrt{\pi}=0.564$ is only 0.8 % away.

**SINDy threshold and scaling.** The threshold acts on raw coefficients. High-degree
monomials of the Lorenz states reach $10^{8}$, so their coefficients are tiny and are cut
by the threshold even when they are spurious; this is why the degree-5 library survives
the noise study as well as degree 2 although the library's condition number grows from
$8\cdot10^{3}$ (degree 2) to $3\cdot10^{10}$ (degree 5). The library size shows up in
the window study instead: the larger library needs a longer trajectory (more of the
attractor) before its columns become distinguishable. Normalizing the library columns
makes the threshold scale-free but needs a retuned threshold.

**Derivatives.** Time derivatives come from second-order central differences of the
(noisy) states, so the noise in $\dot{\mathbf{x}}$ is amplified by $1/\Delta t$; this, not
the regression, sets the noise level at which wrong terms appear.

**Pendulum.** With a polynomial library the fit uses
$-0.598q+0.057q^3-0.112qp^2$: on a single trajectory energy is conserved, so $p^2$ is a
function of $q$ and the polynomial mimics $\sin q$ along the measured orbit only. From a
different initial angle the identified model is wrong by 1.5 rad.

**EUCLID in 1D.** With linear elements, element $e$ contributes $A_e\sigma(\varepsilon_e)
[-1,1]$ to its two nodes. With $\sigma=\sum_k\theta_k g_k(\varepsilon)$ the nodal
internal forces are linear in $\boldsymbol\theta$; the rows are the free nodes (zero
external force), the loaded node (the applied force), and the support (the measured
reaction). No stress is ever used. In a prismatic bar the strain is uniform, so all
free-node rows vanish (up to noise) and one load step gives rank one. Strains come from
differences of noisy displacements and the noise grows with $1/h$, so the mesh is kept
coarse (20 elements).
