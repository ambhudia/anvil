from qiskit_metal import draw, Dict, designs
from qiskit_metal.toolbox_metal import math_and_overrides
from qiskit_metal.qlibrary.core import QComponent


class Feedline(QComponent):
    """
    Simple feedline with lumped port terminal at either end
    """
    component_metadata = Dict(
        short_name='Feedline',
        _qgeometry_table_path='True'
    )
    default_options = Dict(
            origin_x = "0um",
            origin_y = "0um",
            length = "0.2mm",
            trace_gap = "0.02mm",
            trace_width = "0.01mm",
            chip="main",
            layer="1"
    )
    
    TOOLTIP = "Implements feedline"
    
    def make(self):
        p = self.p
        half_outer_len = p.length/2
        half_inner_len = half_outer_len - p.trace_width
        outer_rectangle = draw.LineString(
            [
                [0,-half_outer_len],
                [0,half_outer_len]
            ]
        )
        inner_rectangle = draw.LineString(
            [
                [0,-half_inner_len],
                [0,half_inner_len]
            ]
        )
        pin1 = draw.LineString(
            [
                [0,-half_inner_len],
                [0,-half_inner_len - p.trace_width]
            ]
        )
        pin2 = draw.LineString(
            [
                [0,half_inner_len],
                [0,half_inner_len + p.trace_width]
            ]
        )
        c_items = [outer_rectangle, inner_rectangle, pin1, pin2]
        c_items = draw.translate(c_items, p.origin_x, p.origin_y)
        [outer_rectangle, inner_rectangle, pin1, pin2] = c_items
        

        self.add_qgeometry('path', {'outer_rectangle': outer_rectangle},
                           width=p.trace_width + 2 * p.trace_gap,
                           subtract=True,
                           layer=p.layer)
        self.add_qgeometry('path', {'inner_rectangle': inner_rectangle},
                           width=p.trace_width,
                           layer=p.layer)
        self.add_qgeometry('path', {'pin1': pin1},
                           width=p.trace_width,
                           layer=p.layer)
        self.add_qgeometry('path', {'pin2': pin2},
                           width=p.trace_width,
                           layer=p.layer)