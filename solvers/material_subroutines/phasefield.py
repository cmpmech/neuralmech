"""AT2 phase-field fracture (staggered) as mlhp user subroutines.

`material` is the degraded plane-strain elasticity with the volumetric-deviatoric split
of Amor et al. (2009): only the tensile volumetric and the deviatoric energy are
degraded by g(d) = (1 - d)^2 + k. `damage` is the element integrand of the damage
equation (Gc / l + 2 H) d - Gc l laplace(d) = 2 H with the history H = max psi+. The
history is evaluated in NumPy (`positive_energy`, one value per pixel) or, on refined
meshes, by `history` inside `mlhp.meshFunctionStrainUpdate`.
"""

import numpy as np

from solvers.material_subroutines import build_subroutine

ABI = 1  # mlhp C-interface ABI this routine was written against

# material parameters [E, nu, k] with the residual stiffness k; user fields of
# `material`: [d, E (optional, overrides the constant)], of `damage`: [H, Gc];
# damage data [l] with the length scale l
_SOURCE = r"""
    #include <stdint.h>

    // Plane-strain Voigt [xx, yy, xy] with engineering shear strain.
    static int64_t material(double* stress, double* tangent, double* energyDensity,
                            double* gradient, double* strain, double* xyz, double* rst,
                            double** userFields, double** userData, double* tmp,
                            int64_t* sizes, int64_t* userFieldSizes, int64_t ielement)
    {
        const double* prm = userData[0];

        double E = sizes[2] > 1 ? userFields[1][0] : prm[0];
        double nu = prm[1], k = prm[2];
        double d = userFields[0][0];

        d = d < 0.0 ? 0.0 : ( d > 1.0 ? 1.0 : d );

        double g = ( 1.0 - d ) * ( 1.0 - d ) + k;
        double mu = 0.5 * E / ( 1.0 + nu );
        double K = nu * E / ( ( 1.0 + nu ) * ( 1.0 - 2.0 * nu ) ) + 2.0 * mu / 3.0;

        double trace = strain[0] + strain[1];
        double gK = trace > 0.0 ? g * K : K;

        if(stress)
        {
            stress[0] = gK * trace + 2.0 * g * mu * ( strain[0] - trace / 3.0 );
            stress[1] = gK * trace + 2.0 * g * mu * ( strain[1] - trace / 3.0 );
            stress[2] = g * mu * strain[2];
        }

        if(tangent)
        {
            tangent[0] = tangent[4] = gK + 4.0 * g * mu / 3.0;
            tangent[1] = tangent[3] = gK - 2.0 * g * mu / 3.0;
            tangent[2] = tangent[5] = tangent[6] = tangent[7] = 0.0;
            tangent[8] = g * mu;
        }

        if(energyDensity)
        {
            double dxx = strain[0] - trace / 3.0, dyy = strain[1] - trace / 3.0;
            double deviatoric = dxx * dxx + dyy * dyy + trace * trace / 9.0 +
                                0.5 * strain[2] * strain[2];

            energyDensity[0] = 0.5 * gK * trace * trace + g * mu * deviatoric;
        }

        return 0;
    }

    // Damage equation with one scalar field; matrices are (ndof, ndofpadded).
    static int64_t damage(double** targets, double** shapes, double** mapping,
                          double* rst, double** userFields, double** userData,
                          double* tmp, int64_t* locationMap, int64_t* sizes,
                          int64_t* shapeSizes, int64_t* userFieldSizes,
                          double weight, int64_t ielement)
    {
        int64_t ndof = sizes[3], ndofpadded = sizes[4];

        double H = userFields[0][0], Gc = userFields[1][0], l = userData[0][0];

        const double* N = shapes[0];
        const double* dNdx = shapes[0] + ndofpadded;
        const double* dNdy = shapes[0] + 2 * ndofpadded;

        double reaction = ( Gc / l + 2.0 * H ) * weight, diffusion = Gc * l * weight;

        for(int64_t i = 0; i < ndof; ++i)
        {
            for(int64_t j = 0; j < ndof; ++j)
            {
                targets[0][i * ndofpadded + j] += reaction * N[i] * N[j] +
                    diffusion * ( dNdx[i] * dNdx[j] + dNdy[i] * dNdy[j] );
            }

            targets[1][i] += 2.0 * H * weight * N[i];
        }

        return 0;
    }

    // History update H = max(H, psi+) from the new strain state, for
    // mlhp.meshFunctionStrainUpdate; user field [E (optional)], data [E, nu].
    static int64_t history(double* values, double** gradients, double** strains,
                           double* xyz, double* rst, double** userFields,
                           double** userData, double* tmp, int64_t* sizes,
                           int64_t* userFieldSizes, int64_t ielement)
    {
        const double* prm = userData[0];
        const double* strain = strains[1];

        double E = sizes[3] > 0 ? userFields[0][0] : prm[0];
        double nu = prm[1];
        double mu = 0.5 * E / ( 1.0 + nu );
        double K = nu * E / ( ( 1.0 + nu ) * ( 1.0 - 2.0 * nu ) ) + 2.0 * mu / 3.0;

        double trace = strain[0] + strain[1];
        double dxx = strain[0] - trace / 3.0, dyy = strain[1] - trace / 3.0;
        double deviatoric = dxx * dxx + dyy * dyy + trace * trace / 9.0 +
                            0.5 * strain[2] * strain[2];
        double tension = trace > 0.0 ? trace : 0.0;
        double psi = 0.5 * K * tension * tension + mu * deviatoric;

        values[0] = psi > values[0] ? psi : values[0];

        return 0;
    }

    const unsigned long long material_address = (unsigned long long)&material;
    const unsigned long long damage_address = (unsigned long long)&damage;
    const unsigned long long history_address = (unsigned long long)&history;
"""


def build(tmpdir=None):
    """compile the subroutine; exposes `material_address`, `damage_address` and
    `history_address`."""
    return build_subroutine(
        "_phasefield_subroutine",
        _SOURCE,
        ["material_address", "damage_address", "history_address"],
        tmpdir,
    )


def positive_energy(grad, E, nu):
    """tensile part psi+ of the plane-strain energy (Amor split) from displacement
    gradients (..., 2, 2) and the Young's modulus E (broadcast)."""
    exx, eyy = grad[..., 0, 0], grad[..., 1, 1]
    exy = 0.5 * (grad[..., 0, 1] + grad[..., 1, 0])
    trace = exx + eyy
    mu = 0.5 * E / (1.0 + nu)
    K = nu * E / ((1.0 + nu) * (1.0 - 2.0 * nu)) + 2.0 * mu / 3.0
    deviatoric = (
        (exx - trace / 3) ** 2 + (eyy - trace / 3) ** 2 + trace**2 / 9 + 2 * exy**2
    )
    return 0.5 * K * np.maximum(trace, 0.0) ** 2 + mu * deviatoric
