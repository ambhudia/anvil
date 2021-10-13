import numpy as np
from qiskit_metal import draw, Dict, designs
from qiskit_metal.toolbox_metal import math_and_overrides
from qiskit_metal.qlibrary.core import QComponent

class BendedEnds(QComponent):
    component_metadata = Dict(
        short_name='BendedEnds',
        _qgeometry_table_path='True'
    )
    default_options = Dict(
            origin_x = "0um",
            origin_y = "0um",
            x_length = "0.2mm",
            trace_gap = "0.01mm",
            trace_width = "0.01mm",
            meander_radius = "0.5mm",
            n_points_meander = 128,
            orientation = True,
            straight_first = True, 
            chip="main",
            layer="1"
    )
    
    TOOLTIP = "Half radius and straight edge on end"
    
    def make(self):
        
        p = self.p
        origin_x, origin_y = p.origin_x, p.origin_y
        n_points_meander = p.n_points_meander
        theta = np.pi/2
        d_theta = theta/n_points_meander
        orientation = p.orientation
        straight_first = p.straight_first
        meander_radius = p.meander_radius
        
        if orientation:
            if p.straight_first:
                # quadrant 1
                angles = np.arange(theta, 0, -d_theta)
                xs = meander_radius * np.cos(angles)
                ys = meander_radius * np.sin(angles) - meander_radius
            else:
                # quadrant 2
                angles = np.arange(2*theta, theta, -d_theta)
                xs = meander_radius * np.cos(angles) + meander_radius
                ys = meander_radius * np.sin(angles)
        else:
            if p.straight_first:
                # quadrant 3
                angles = np.arange(2*theta, 3*theta, d_theta)
                xs = meander_radius * np.cos(angles) + meander_radius
                ys = meander_radius * np.sin(angles)
            else:
                # quadrant 4
                angles = np.arange(3*theta, 4*theta, d_theta)
                xs = meander_radius * np.cos(angles)
                ys = meander_radius * np.sin(angles) + meander_radius
        
        if p.straight_first:
            x_pts, y_pts = [np.array([origin_x])], [np.array([origin_y])]
            x, y = origin_x + p.x_length, origin_y
            x_pts.append(xs+x)
            y_pts.append(ys+y)
        else:
            x_pts, y_pts = [xs+origin_x], [ys+origin_y]

            x, y = xs[-1] + origin_x + p.x_length, ys[-1] + origin_y
            x_pts.append(np.array([x]))
            y_pts.append(np.array([y]))
        
        x_pts = np.concatenate(x_pts)
        y_pts = np.concatenate(y_pts)
        self.points = np.array([x_pts, y_pts]).T
        self.qgeometry_table_usage = {'path': True, 'poly': False, 'junction': False}
        self.make_elements(self.points)
        self.x_end = x_pts[-1]
        self.y_end = y_pts[-1]
        self.x_pts = x_pts
        self.y_pts = y_pts
        
        
    def make_elements(self, pts: np.ndarray):
        """Turns the CPW points into design elements, and add them to the
        design object.

        Args:
            pts (np.ndarray): Array of points
        """
        p = self.p
        # prepare the routing track
        line = draw.LineString(pts)

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
        
    def get_endpoints(self):
        return self.x_end, self.y_end
    
    
    def get_points(self):
        return self.x_pts, self.y_pts