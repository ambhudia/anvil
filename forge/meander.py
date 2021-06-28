import math
import numpy as np
from qiskit_metal import draw, Dict, designs
from qiskit_metal.toolbox_metal import math_and_overrides
from qiskit_metal.qlibrary.core import QComponent


class Meander(QComponent):
    """
    Meandered CPW inductor
    
    To simplify the process, I utilise the same base code used to define
    the meanders in Klayout, albeit heavly modified.
    """
    component_metadata = Dict(
        short_name='Meandered CPW inductor',
        _qgeometry_table_path='True'
    )
    default_options = Dict(
            origin_x = "0um",
            origin_y = "0um",
            n_points_meander = "128", 
            width = "5um",
            width_incl_meanders = "true",
            spacing = "1um",
            n_lines = 5,
            meander_radius = "0.5um",
            coupler_length = "5um",
            coupler_spacing = "2um",
            coupler_radius = "1um",
            n_points_coupler = "256",
            coupler_orientation = "true",
            trace_gap = "0.02um",
            trace_width = "0.01um",
            end_length = "5um",
            pin_width = "0.02um",
    )
    

    TOOLTIP = """Implements a meandered CPW inductor"""
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
        """The make function implements the logic that creates the geometry
        (poly, path, etc.) from the qcomponent.options dictionary of
        parameters, and the adds them to the design, using
        qcomponent.add_qgeometry(...), adding in extra needed information, such
        as layer, subtract, etc."""
        
        orientation = True
        
        x_pts = []
        y_pts = []
        x_meanders = []
        y_meanders = []
        
        ############################## COUPLER ##############################
        # Create the coupler 
        x = 0
        y = self.p.pin_width
        
        x_pts.append(np.array([x]))
        y_pts.append(np.array([y]))
        
        x = 0
        y = self.p.coupler_length
        
        x_pts.append(np.array([x]))
        y_pts.append(np.array([y]))
        
        
        # Create bend
        x_meander, y_meander = self.bend(
            x, 
            y, 
            self.p.n_points_coupler, 
            orientation,
            True
        )
        
        x_pts.append(x_meander)
        y_pts.append(y_meander)
        
        x -= self.p.coupler_spacing
        x_pts.append(np.array([x]))
        y_pts.append(np.array([y]))
        
        orientation = not orientation
        
        
        ############################## MAIN BODY #############################
        
        if self.p.width_incl_meanders:
            cutoff = self.get_cutoff()
        x += self.p.spacing
        for line in range(self.p.n_lines-1):            
            y_dist = self.p.width
            
            
            if self.p.width_incl_meanders:
                if line == 0:
                    y_dist -= cutoff
                elif line == self.p.n_lines-1:
                    y_dist -= cutoff
                else:
                    y_dist -= 2*cutoff
                if orientation:
                    y += y_dist
                else: 
                    y -= y_dist
            x -= self.p.spacing
            x_pts.append(np.array([x]))
            y_pts.append(np.array([y]))
        
            
            meander_x, meander_y = self.bend(
                x, 
                y, 
                self.p.n_points_meander, 
                orientation
            )
            x_pts.append(meander_x)
            y_pts.append(meander_y)
            orientation =  not orientation
        
        ################################### End Length #########################
        y_dist = self.p.end_length - self.p.pin_width
            
        x -= self.p.spacing    
        if self.p.width_incl_meanders:
            y_dist -= cutoff
            if orientation:
                y += y_dist
            else: 
                y -= y_dist
        x_pts.append(np.array([x]))
        y_pts.append(np.array([y]))
        
        ################################ Create Terminations ####################
        sign = 1 if self.p.coupler_orientation else -1
        pin1 = np.zeros([2, 2])
        pin1[0][0], pin1[1][0] = self.p.origin_x, self.p.origin_x
        pin1[0][1], pin1[1][1] = self.p.origin_y, sign * self.p.pin_width + self.p.origin_y
        self.make_elements(pin1, True)


        sign = sign if self.p.n_lines%2 == 0 else -1*sign
        pin2 = np.zeros([2, 2])
        pin2[0][0], pin2[1][0] = x + self.p.origin_x, x + self.p.origin_x
        pin2[0][1], pin2[1][1] = y + self.p.origin_y, y + self.p.origin_y + sign*self.p.pin_width
        self.make_elements(pin2, True)
        
        ############################### Concatenate Arrays #####################
        x_pts = np.concatenate(x_pts)
        y_pts = np.concatenate(y_pts)
        
        
        # Set points
        self.points = np.array([x_pts, y_pts]).T
        
        
        # Transform the points
        if not self.p.coupler_orientation:
            self.points = -self.points
        
        self.points[:,0] += self.p.origin_x
        self.points[:,1] += self.p.origin_y
        # Make points into elements
        self.qgeometry_table_usage = {'path': True, 'poly': False, 'junction': False}
        self.make_elements(self.points)
    
    
    def get_theta(self, is_coupler:bool) -> float:
        """
           .|.     /
         .  |  .  /  
        .___|___./
        |   |   /|  i.e. the angle marked x 
        |   |  / |
            |x/ 
            |/  
   
        This is useful in the event that the meander radius is greater
        than half the spacing between the vertical segments of the inductor
        """
        if is_coupler:
            theta = math.asin(0.5*self.p.coupler_spacing/self.p.coupler_radius)
        else:
            theta = math.asin(0.5*self.p.spacing/self.p.meander_radius)
        return theta
        
    def get_cutoff(self, is_coupler:bool = False) -> float:
        """
        Get the height of a bend            
        
           . .    <-
         .     .    } i.e. this
        .       . <-
        
        This is useful in the event that the meander radius is greater
        than half the spacing between the vertical segments of the inductor
        """
        theta = 1-math.cos(self.get_theta(is_coupler))
        if is_coupler:
            return self.p.coupler_radius * theta
        else:
            return self.p.meander_radius * theta


    def bend(self, x, y, n_pts, orientation, is_coupler=False):
        """
        Generate the points needed to create one of the bent segments
        
           . .   
         .     .     i.e. one of these
        .       . 
        """
        ############## Generate an array of angles for the arc #############
        n_pts+=1
        theta = self.get_theta(is_coupler)
        if is_coupler:
            d_theta = 2 * theta/self.p.n_points_coupler
        else:
            d_theta = 2 * theta/self.p.n_points_meander
            
        current_angle = d_theta

        angles = np.arange(-theta, theta, d_theta)

        if n_pts%2 == 0: # truncate extraneous entries
            angles = angles[1:]
        else:
            angles = angles[:-1]
        #####################################################################
        
        if is_coupler:
            y_init = self.p.coupler_radius * math.cos(theta)
            y_arr = self.p.coupler_radius * np.cos(angles) - y_init
            x_arr = x - (
                self.p.coupler_radius * np.sin(angles) + self.p.coupler_spacing/2
            )
        else:
            y_init = self.p.meander_radius * math.cos(theta)
            y_arr = self.p.meander_radius * np.cos(angles) - y_init
            x_arr = x - (
                self.p.meander_radius * np.sin(angles) + self.p.spacing/2
            )
        if orientation:
            y_arr += y
        else:
            y_arr = y-y_arr
        return x_arr, y_arr[::-1]
    
    def plot(self):
        plt.plot(self.points[:,0],self.points[:,1])
        
    def get_pins(self):
        return self.start_pin, self.end_pin
    
    def make_elements(self, pts: np.ndarray, short: bool = False):
        """Turns the CPW points into design elements, and add them to the
        design object.

        Args:
            pts (np.ndarray): Array of points
        """
        p = self.p
        # prepare the routing track
        line = draw.LineString(pts)

        # expand the routing track to form the substrate core of the cpw
        if not short:
            self.add_qgeometry(
                'path', {'trace': line},
                width=p.trace_width,
                layer=p.layer
                )

        self.add_qgeometry('path', {'cut': line},
                               width=p.trace_width + 2 * p.trace_gap,
                               layer=p.layer,
                               subtract=True)
