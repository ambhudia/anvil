"""Draw a CPW geometry given a series of x, y points
"""
import numpy as np
from qiskit_metal import draw, Dict, designs
from qiskit_metal.toolbox_metal import math_and_overrides
from qiskit_metal.qlibrary.core import QComponent


class CPWMerged(QComponent):

    component_metadata = Dict(
        short_name='Merged CPWs',
        _qgeometry_table_path='True'
    )
    default_options = Dict(
            x_pts = None,
            y_pts = None,
            trace_gap = "0.02um",
            trace_width = "0.01um",
    )

    def __init__(self,
                 design,
                 name: str = None,
                 options: Dict = None,
                 type: str = "CPW",
                 **kwargs):


        self.type = type.upper().strip()
        
        # regular QComponent boot, including the run of make()
        super().__init__(design, name, options, **kwargs)

    def make(self):
        p = self.p
        x_pts, y_pts = p.x_pts, p.y_pts
        self.points = np.array([x_pts, y_pts]).T
        self.qgeometry_table_usage = {'path': True, 'poly': False, 'junction': False}
        
        # prepare the routing track
        line = draw.LineString(self.points)

        # expand the routing track to form the substrate core of the cpw
        self.add_qgeometry(
            'path', {'trace': line},
            width=p.trace_width,
            layer=p.layer,
            fillet=0
            )

        self.add_qgeometry('path', {'cut': line},
                               width=p.trace_width + 2 * p.trace_gap,
                               layer=p.layer,
                               subtract=True,
                               fillet=0
                               )
    def get_points(self):
            return self.p.x_pts, self.p.y_pts