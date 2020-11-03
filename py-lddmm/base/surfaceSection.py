from .curves import Curve


class SurfaceSection:
    def __init__(self, curve=None, affine=None):
        self.curve = Curve(curve)
        if affine is None:

