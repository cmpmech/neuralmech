"""cubic-stiffening plane-stress elasticity as an mlhp user-material subroutine.

The law derives from the potential W(eps) = 1/2 eps.C0.eps + c/4 (eps.eps)^2, so

    stress = C0 eps + c (eps.eps) eps
    tangent = C0 + c (eps.eps) I + 2 c eps (outer) eps

with eps the Voigt strain [exx, eyy, gxy] (engineering shear) and C0 the isotropic
plane-stress stiffness from [E, nu]. It is path-independent, so it carries no history
and is registered with incremental=False (the callback receives the total strain).
"""

from solvers.material_subroutines import build_subroutine

ABI = 1  # mlhp C-interface ABI this routine was written against

_SOURCE = r"""
    #include <stdint.h>

    // 2D plane-stress nonlinear-elastic law; userData[0] = [E, nu, c].
    static int64_t material(double* stress, double* tangent, double* energyDensity,
                            double* gradient, double* strain, double* xyz, double* rst,
                            double** userFields, double** userData, double* tmp,
                            int64_t* sizes, int64_t* userFieldSizes, int64_t ielement)
    {
        double E = userData[0][0], nu = userData[0][1], c = userData[0][2];
        double f = E / (1.0 - nu * nu);

        double C0[9] = { f,      f * nu, 0.0,
                         f * nu, f,      0.0,
                         0.0,    0.0,    f * (1.0 - nu) / 2.0 };

        double ee = strain[0] * strain[0] + strain[1] * strain[1] + strain[2] * strain[2];

        if(stress) for(int i = 0; i < 3; ++i)
        {
            stress[i] = c * ee * strain[i];

            for(int j = 0; j < 3; ++j) stress[i] += C0[i * 3 + j] * strain[j];
        }

        if(tangent) for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                tangent[i * 3 + j] = C0[i * 3 + j] + 2.0 * c * strain[i] * strain[j]
                                     + (i == j ? c * ee : 0.0);

        return 0;
    }

    const unsigned long long material_address = (unsigned long long)&material;
"""


def build(tmpdir=None):
    """compile the subroutine; exposes `material_address`."""
    return build_subroutine(
        "_cubic_stiffening_subroutine", _SOURCE, ["material_address"], tmpdir
    )
