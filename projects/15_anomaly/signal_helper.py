import math


# --------------------------------- bump ---------------------------------
def bump(t, t0, width, height):
    if abs(t - t0) < width:
        return (
            math.exp(1.0 / (((t - t0) / width) ** 2 - 1)) * math.exp(1) * (height - 1)
            + 1
        )
    else:
        return 1


# ----------------------------- degrading k ------------------------------
def get_degrading_k(bump_center, width, height, k_base, index):
    def degrade(t):
        if t < bump_center:
            return bump(t, bump_center, width, height)
        else:
            return height

    def k(t):
        k_mod = k_base.copy()  # assumes k is a fixed vector
        k_mod[index] *= degrade(t)
        return k_mod

    return k


# ------------------------------ spiking f -------------------------------
def get_spiking_f(bump_center, width, height, f_base, index):
    amp = lambda t: bump(t, bump_center, width, height)
    return lambda t: [
        fi * amp(t) if i == index else fi for i, fi in enumerate(f_base(t))
    ]
