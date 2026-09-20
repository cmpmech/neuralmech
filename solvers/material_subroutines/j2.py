"""small-strain J2 (von Mises) plasticity as an mlhp user-material subroutine.

History per material point is [stress (6), backstress (6), equivalent plastic strain]
in Voigt order [xx, yy, zz, xy, yz, xz] with engineering shear. The callbacks expand
the 3-component plane-strain state to the 6-component routine and reduce the
consistent tangent back to the in-plane [xx, yy, xy] block.
"""

from solvers.material_subroutines import build_subroutine

ABI = 1  # mlhp C-interface ABI this routine was written against
NHISTORY = 13  # doubles per material point

# material parameter layout shared by both callbacks: [E, nu, sigmaY, H, beta]
# with hardening modulus H and beta in [0, 1] (0 = isotropic, 1 = kinematic).
_SOURCE = r"""
    #include <math.h>
    #include <stdint.h>
    #include <string.h>

    // 3D small-strain J2 return mapping with isotropic/kinematic hardening.
    static void j2(const double* prm, const double* history0, const double* dstrain,
                   double* stress, double* tangent, double* history1)
    {
        double E = prm[0], nu = prm[1], sigmaY = prm[2], H = prm[3], beta = prm[4];

        double lambda = nu * E / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double mu = 0.5 * E / (1.0 + nu);

        double C[36] = { 0.0 };

        C[0 * 6 + 0] = C[1 * 6 + 1] = C[2 * 6 + 2] = lambda + 2.0 * mu;
        C[0 * 6 + 1] = C[0 * 6 + 2] = C[1 * 6 + 0] = lambda;
        C[1 * 6 + 2] = C[2 * 6 + 0] = C[2 * 6 + 1] = lambda;
        C[3 * 6 + 3] = C[4 * 6 + 4] = C[5 * 6 + 5] = mu;

        // Trial stress
        double trial[6];

        for(int i = 0; i < 6; ++i)
        {
            trial[i] = history0[i];

            for(int j = 0; j < 6; ++j)
            {
                trial[i] += C[i * 6 + j] * dstrain[j];
            }
        }

        // Shifted deviatoric stress
        double trace = trial[0] + trial[1] + trial[2];
        double eta[6];

        for(int i = 0; i < 6; ++i)
        {
            eta[i] = trial[i] - history0[6 + i] - (i < 3 ? trace / 3.0 : 0.0);
        }

        double etaNorm = sqrt(eta[0] * eta[0] + eta[1] * eta[1] + eta[2] * eta[2] +
                       2.0 * (eta[3] * eta[3] + eta[4] * eta[4] + eta[5] * eta[5]));

        double ep0 = history0[12];
        double yield = etaNorm - sqrt(2.0 / 3.0) * (sigmaY + (1.0 - beta) * H * ep0);

        if(yield < 0.0)
        {
            memcpy(stress, trial, 6 * sizeof(double));
            memcpy(tangent, C, 36 * sizeof(double));
            memcpy(history1, history0, 13 * sizeof(double));
            memcpy(history1, stress, 6 * sizeof(double));

            return;
        }

        // Return mapping
        double dlambda = yield / (2.0 * mu + 2.0 / 3.0 * H);
        double n[6];

        for(int i = 0; i < 6; ++i)
        {
            n[i] = eta[i] / etaNorm;
            stress[i] = trial[i] - 2.0 * mu * dlambda * n[i];
        }

        // Algorithmic tangent: elastic with plastic correction and deviatoric projection
        double c1 = 4.0 * mu * mu / (2.0 * mu + 2.0 / 3.0 * H);
        double c2 = 4.0 * mu * mu * dlambda / etaNorm;

        for(int i = 0; i < 6; ++i)
        {
            for(int j = 0; j < 6; ++j)
            {
                tangent[i * 6 + j] = C[i * 6 + j] - (c1 - c2) * n[i] * n[j];
            }
        }

        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                tangent[i * 6 + j] += c2 / 3.0;
            }

            tangent[i * 6 + i] -= c2;
            tangent[(i + 3) * 6 + (i + 3)] -= c2 / 2.0;
        }

        memcpy(history1, stress, 6 * sizeof(double));

        for(int i = 0; i < 6; ++i)
        {
            history1[6 + i] = history0[6 + i] + 2.0 / 3.0 * beta * H * dlambda * n[i];
        }

        history1[12] = ep0 + sqrt(2.0 / 3.0) * dlambda;
    }

    // Plane-strain Voigt indices into the 6-component routine: xx, yy, xy.
    static const int M[3] = { 0, 1, 3 };

    // Constitutive callback: strain is the increment (incremental=True),
    // userFields[0] the history, userData[0] the parameter vector.
    static int64_t material(double* stress, double* tangent, double* energyDensity,
                            double* gradient, double* strain, double* xyz, double* rst,
                            double** userFields, double** userData, double* tmp,
                            int64_t* sizes, int64_t* userFieldSizes, int64_t ielement)
    {
        double dstrain[6] = { 0.0 }, stressLocal[6], tangentLocal[36], history1[13];

        for(int a = 0; a < 3; ++a) dstrain[M[a]] = strain[a];

        j2(userData[0], userFields[0], dstrain, stressLocal, tangentLocal, history1);

        if(stress) for(int a = 0; a < 3; ++a) stress[a] = stressLocal[M[a]];

        if(tangent) for(int a = 0; a < 3; ++a)
                        for(int b = 0; b < 3; ++b)
                            tangent[a * 3 + b] = tangentLocal[M[a] * 6 + M[b]];

        return 0;
    }

    // History update callback: commits the return mapping once per converged
    // load step. Receives the total strains of both states.
    static int64_t update(double* history, double** gradients, double** strains,
                          double* xyz, double* rst, double** userFields, double** userData,
                          double* tmp, int64_t* sizes, int64_t* userFieldSizes,
                          int64_t ielement)
    {
        double dstrain[6] = { 0.0 }, stress[6], tangent[36], history1[13];

        for(int a = 0; a < 3; ++a) dstrain[M[a]] = strains[1][a] - strains[0][a];

        j2(userData[0], history, dstrain, stress, tangent, history1);

        memcpy(history, history1, 13 * sizeof(double));

        return 0;
    }

    const unsigned long long material_address = (unsigned long long)&material;
    const unsigned long long update_address = (unsigned long long)&update;
"""


def build(tmpdir=None):
    """compile the subroutine; exposes `material_address` and `update_address`."""
    return build_subroutine(
        "_j2_subroutine", _SOURCE, ["material_address", "update_address"], tmpdir
    )
