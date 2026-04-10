import math

def bump(t, t0, width, height):
    if abs(t - t0) < width:
        return math.exp(1. / (((t - t0) / width)**2 - 1)) * math.exp(1) * (height - 1) + 1
    else:
        return 1
