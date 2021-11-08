from enum import Enum
import numpy as np

class BendedEndTypes(Enum):
    STRAIGHT_DOWN = (True , True)
    UP_STRAIGHT   = (True , False)
    STRAIGHT_UP   = (False, True)
    DOWN_STRAIGHT = (False, False)


class BendedEnd:
    """Generate x, y points for the bended ends
    """
    def __init__(self):
        self.origin_x = None
        self.origin_y = None
        self.x_length = None
        self.n_points_meander = None
        self.orientation, self.straight_first = None, None
        self.meander_radius = None

    def generate_bend(
        self,         
        origin_x, 
        origin_y, 
        n_points_meander, 
        type_,
        meander_radius, 
        x_length
        ):

        self.origin_x = origin_x
        self.origin_y = origin_y
        self.x_length = x_length
        self.n_points_meander = n_points_meander
        self.orientation, self.straight_first = type_
        self.meander_radius = meander_radius

        p = self
        origin_x, origin_y = p.origin_x, p.origin_y
        n_points_meander = p.n_points_meander
        theta = np.pi/2
        d_theta = theta/n_points_meander
        orientation = p.orientation
        straight_first = p.straight_first
        meander_radius = p.meander_radius

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
        y_pts = y_pts - y_pts[0]
        x_pts = x_pts - x_pts[0] + origin_x
        if not orientation:
            y_pts=-y_pts
        y_pts += origin_y
        return x_pts, y_pts

class GenericResonatorSection:
    def generate_semicircle(self, r, x0, y0, start, end, n_pts, orientation):
        theta = np.linspace(start, end, n_pts)

        x_pts = r * np.cos(theta)
        y_pts = r * np.sin(theta)
        x_pts = -x_pts

        if not orientation:
            y_pts = -y_pts

        x_pts += x0 - x_pts[0]
        y_pts += y0 - y_pts[0]

        return x_pts, y_pts
    
    def max_n_first_section(self, L, c, l):
        n = 0
        while True:
            L_rem = L - n * l - n * c
            if L_rem < 0:
                break
            n += 1
        return n - 1

    def generate_resonator(
        self, 
        x,
        y,
        L2,
        l2,
        r2,
        npts,
        n_pts=None,
        carryover=None,
        min_angle=None,
        upwards=False,
    ):
        c2 = np.pi*r2
        x_pts, y_pts = [], []
        x_pts.append(x)
        y_pts.append(y)

        if carryover:
            L2 -= carryover
            if L2 < 0:
                raise Exception("Check parameters. Straights too large.")
            # remaining straight section
            if upwards:
                y = y + carryover
            else:
                y = y - carryover

            carryover = None
            x_pts.append(x)
            y_pts.append(y)
            straight_first = False

        elif min_angle:
            L2 -= (np.pi - min_angle) * r2
            if L2 < 0:
                raise Exception("Check parameters. Bends too large.")

            xs, ys = self.generate_semicircle(
                r2, x, y, min_angle, np.pi, n_pts, orientation=upwards
            )

            x_pts.extend(xs.tolist())
            y_pts.extend(ys.tolist())
            x = x_pts[-1]
            y = y_pts[-1]

            upwards = not upwards
            min_angle = None
            straight_first = True

        n = self.max_n_first_section(L2, c2, l2)

        if straight_first:
            for _ in range(n):
                if upwards:
                    y = y + l2
                else:
                    y = y - l2
                L2 -= l2
                x_pts.append(x)
                y_pts.append(y)

                xs, ys = self.generate_semicircle(r2, x, y, 0, np.pi, npts, orientation=upwards)
                x_pts.extend(xs.tolist())
                y_pts.extend(ys.tolist())
                x = x_pts[-1]
                y = y_pts[-1]
                L2 -= c2
                upwards = not upwards
        else:
            for _ in range(n):
                xs, ys = self.generate_semicircle(r2, x, y, 0, np.pi, npts, orientation=upwards)
                x_pts.extend(xs.tolist())
                y_pts.extend(ys.tolist())
                x = x_pts[-1]
                y = y_pts[-1]
                L2 -= c2
                upwards = not upwards

                if upwards:
                    y = y + l2
                else:
                    y = y - l2
                L2 -= l2
                x_pts.append(x)
                y_pts.append(y)

        if straight_first:
            if L2 >= l2:
                l = l2
                L2 -= l2
            else:
                l = L2
                carryover = l2 - L2
                L2 -= L2
            if upwards:
                y = y + l
            else:
                y = y - l

            x_pts.append(x)
            y_pts.append(y)

            if L2 > 0:
                max_angle = np.pi * L2 / c2
                n_pts = int(max_angle / np.pi * npts)

                L2 -= max_angle * r2

                xs, ys = self.generate_semicircle(
                    r2, x, y, 0, max_angle, n_pts, orientation=upwards
                )

                x_pts.extend(xs.tolist())
                y_pts.extend(ys.tolist())
                x = x_pts[-1]
                y = y_pts[-1]

                if n_pts != npts:
                    min_angle = max_angle
                    n_pts = npts - n_pts
                else:
                    min_angle = None
                    n_pts = None

        else:
            if L2 < c2:
                max_angle = np.pi * L2 / c2
                L2 = 0
            else:
                max_angle = np.pi
                L2 -= max_angle * r2
            n_pts = int(max_angle / np.pi * npts)

            xs, ys = self.generate_semicircle(r2, x, y, 0, max_angle, n_pts, orientation=upwards)

            x_pts.extend(xs.tolist())
            y_pts.extend(ys.tolist())
            x = x_pts[-1]
            y = y_pts[-1]

            if n_pts != npts:
                min_angle = max_angle
                n_pts = npts - n_pts

            else:
                upwards = not upwards
                min_angle = None
                n_pts = None

            if L2 > 0:
                carryover = l2 - L2
                l = l2 - carryover
                L2 -= l
                if carryover < 0:
                    raise Exception("Check logic")

                if upwards:
                    y = y + l
                else:
                    y = y - l

                x_pts.append(x)
                y_pts.append(y)

        if L2 > 1e-15:
            raise Exception(f"L2 = {L2}!!!!")

        return x_pts, y_pts, upwards, x, y, min_angle, n_pts, carryover


def test_length(resonator, L, percent_err_tol):
    L_ = 0
    x, y = resonator
    x, y = np.array(x), np.array(y)
    x1, y1 = x[:-1], y[:-1]
    x2, y2 = x[1:], y[1:]

    for (xi, xf, yi, yf) in zip(x1, x2, y1, y2):
        dist = np.sqrt((xf - xi) ** 2 + (yf - yi) ** 2)
        L_ += dist

    err = abs(L - L_) / L * 100
    if err > percent_err_tol:
        print(err)
        return False
    else:
        return True


def rotate(x, y, angle):
    angle = np.deg2rad(angle)
    x0, y0 = x[0], y[0]
    x -= x0
    y -= y0
    xnew = x*np.cos(angle)-y*np.sin(angle)
    ynew = x*np.sin(angle)+y*np.cos(angle)
    xnew += x0
    ynew += y0
    return xnew, ynew

def generate_series_resonator_params(L, r, x, w, s):
    n = 1
    l = 1
    
    while(l >0):
        l = 1/n * (L-(n+1)*np.pi*r-2*x+2*r)
        n += 1
    ns = np.array([i for i in range(1, n)], dtype=int)
    ls = np.asarray([1/n * (L-(n+1)*np.pi*r-2*x+2*r) for n in ns])
    Ws = 2*r*(ns+1)+2*x
    Hs = ls+2*r+w+2*s
    As = Ws * Hs
    return ns, ls, Ws, Hs, As


def remove_duplicates(x, y):
    """Remove pairwise duplicate x, y points to prevent backtracking
    """
    coords = np.column_stack((x, y))
    coords = np.unique(coords, axis=0)
    x, y = coords[:, 0], coords[:, 1]
    return x, y


class BendedEndResonator:   
    def generate_resonator(
        self,
        origin_x, 
        origin_y, 
        N, 
        L, 
        r, 
        l, 
        x_length,
        upwards = False,
        n_points_meander = 3600,
        clean=True
        ):
        insert_x, insert_y  = [], []
        x_pts, y_pts = [], []
        # length to autogenerate The rest is done manually, as seen below.
        _L = L - 2*x_length - np.pi*r
        carryover = l/2 - r

        # set type of bend
        if upwards:
            type_ = BendedEndTypes.STRAIGHT_UP.value
        else:
            type_ = BendedEndTypes.STRAIGHT_DOWN.value

        # generate first stright-bend
        BE = BendedEnd()
        xs, ys = BE.generate_bend(
            origin_x,
            origin_y,
            int(n_points_meander/2),
            type_,
            r, 
            x_length
            )
        x_pts.append(xs)
        y_pts.append(ys)

        x, y = xs[-1], ys[-1]
        
        if N == 1:
            GRS = GenericResonatorSection()
            (xs, ys, upwards, _ , _ , _ ,_ , _ ) = GRS.generate_resonator(
                x, 
                y, 
                _L, 
                l,
                r,
                n_points_meander,
                carryover = carryover,
                upwards = upwards
            )
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
        if N == 2:
            GRS = GenericResonatorSection()
            (xs, ys, upwards, _ , _ , _ ,_ , _ ) = GRS.generate_resonator(
                x, 
                y, 
                _L, 
                l,
                r,
                n_points_meander,
                carryover = carryover,
                upwards = upwards
            )
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            
            if upwards:
                type_ = BendedEndTypes.UP_STRAIGHT.value
            else:
                type_ = BendedEndTypes.DOWN_STRAIGHT.value


            BE = BendedEnd()
            xs, ys = BE.generate_bend(
                x,
                y,
                int(n_points_meander/2),
                type_,
                r, 
                x_length
                )
            
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            insert_x.append(x)
            insert_y.append(y)
            if upwards:
                type_ = BendedEndTypes.STRAIGHT_DOWN.value
            else:
                type_ = BendedEndTypes.STRAIGHT_UP.value


            BE = BendedEnd()
            xs, ys = BE.generate_bend(
                x,
                y,
                int(n_points_meander/2),
                type_,
                r, 
                x_length
                )
            
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]

            upwards = not upwards ## ! caution
            
            GRS = GenericResonatorSection()
            (xs, ys, upwards, _ , _ , _ ,_ , _ ) = GRS.generate_resonator(
                x, 
                y, 
                _L, 
                l,
                r,
                n_points_meander,
                carryover = carryover,
                upwards = upwards
            )
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            
        if N == 3:
            GRS = GenericResonatorSection()
            (xs, ys, upwards, _ , _ , _ ,_ , _ ) = GRS.generate_resonator(
                x, 
                y, 
                _L, 
                l,
                r,
                n_points_meander,
                carryover = carryover,
                upwards = upwards
            )
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            
            if upwards:
                type_ = BendedEndTypes.UP_STRAIGHT.value
            else:
                type_ = BendedEndTypes.DOWN_STRAIGHT.value


            BE = BendedEnd()
            xs, ys = BE.generate_bend(
                x,
                y,
                int(n_points_meander/2),
                type_,
                r, 
                x_length
                )
            
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            insert_x.append(x)
            insert_y.append(y)

            if upwards:
                type_ = BendedEndTypes.STRAIGHT_DOWN.value
            else:
                type_ = BendedEndTypes.STRAIGHT_UP.value

            BE = BendedEnd()
            xs, ys = BE.generate_bend(
                x,
                y,
                int(n_points_meander/2),
                type_,
                r, 
                x_length
                )
            
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            
            upwards = not upwards
            GRS = GenericResonatorSection()
            (xs, ys, upwards, _ , _ , _ ,_ , _ ) = GRS.generate_resonator(
                x, 
                y, 
                _L, 
                l,
                r,
                n_points_meander,
                carryover = carryover,
                upwards = upwards
            )
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]

            if upwards:
                type_ = BendedEndTypes.UP_STRAIGHT.value
            else:
                type_ = BendedEndTypes.DOWN_STRAIGHT.value


            BE = BendedEnd()
            xs, ys = BE.generate_bend(
                x,
                y,
                int(n_points_meander/2),
                type_,
                r, 
                x_length
                )
            
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            insert_x.append(x)
            insert_y.append(y)

            if upwards:
                type_ = BendedEndTypes.STRAIGHT_DOWN.value
            else:
                type_ = BendedEndTypes.STRAIGHT_UP.value

            BE = BendedEnd()
            xs, ys = BE.generate_bend(
                x,
                y,
                int(n_points_meander/2),
                type_,
                r, 
                x_length
                )
            
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]
            upwards = not upwards
            GRS = GenericResonatorSection()
            (xs, ys, upwards, _ , _ , _ ,_ , _ ) = GRS.generate_resonator(
                x, 
                y, 
                _L, 
                l,
                r,
                n_points_meander,
                carryover = carryover,
                upwards = upwards
            )
            x_pts.append(xs)
            y_pts.append(ys)
            x, y = xs[-1], ys[-1]

            ###########################################3
        if upwards:
            type_ = BendedEndTypes.UP_STRAIGHT.value
        else:
            type_ = BendedEndTypes.DOWN_STRAIGHT.value


        BE = BendedEnd()
        xs, ys = BE.generate_bend(
            x,
            y,
            int(n_points_meander/2),
            type_,
            r, 
            x_length
            )
        
        x_pts.append(xs)
        y_pts.append(ys)
    
        x_pts = np.concatenate(x_pts)
        y_pts = np.concatenate(y_pts)
        
        if clean:
            x_pts, y_pts = remove_backtracking_points(x_pts, y_pts, name="Series")
        
        return x_pts, y_pts, insert_x, insert_y


def generate_shunt_resonator_params(W, w, s, r, x):
    l = W-(w+2*s)-2*r
    carryover1 = l-r-x
    carryover2 = x-2*r
    carryover = max(carryover1, carryover2)
    return l, carryover


class ShuntResonator:
    def generate_resonator(
        self, 
        origin_x, 
        origin_y, 
        W, 
        w, 
        s, 
        r,
        L, 
        x_length, 
        leftward = True,
        above = True,
        n_points_meander = 3600,
        clean=True
         ):
        l, carryover = generate_shunt_resonator_params(W, w, s, r, origin_x)
        _L = L-w/2-x_length-np.pi/2*r

        x_pts, y_pts = [np.array([0])], [np.array([0])]
        x, y = x_pts[-1], y_pts[-1]

        x_pts.append(x+x_length)
        y_pts.append(y)
        x, y = x_pts[-1], y_pts[-1]

        upwards = False

        type_ = BendedEndTypes.STRAIGHT_DOWN.value
        BE = BendedEnd()
        xs, ys = BE.generate_bend(
            x[0],
            y[0],
            int(n_points_meander/2),
            type_,
            r, 
            0
            )
        x_pts.append(xs)
        y_pts.append(ys)

        x, y = xs[-1], ys[-1]
        GRS = GenericResonatorSection()
        (xs, ys, _, _ , _ , _ ,_ , _) = GRS.generate_resonator(
                x, 
                y, 
                _L, 
                l,
                r,
                n_points_meander,
                carryover = carryover,
                upwards = upwards
            )
        x_pts.append(xs)
        y_pts.append(ys)
        x_pts = np.concatenate(x_pts)
        y_pts = np.concatenate(y_pts)
        
        if clean:
            x_pts, y_pts = remove_backtracking_points(x_pts, y_pts, name="Shunt")
        
        above = not above 
        if above:
            if not leftward:
                y_pts = -y_pts
        else:
            if not leftward:
                y_pts = -y_pts

        x_pts+=origin_x
        y_pts+=origin_y

        if above:
            x_pts, y_pts = rotate(x_pts, y_pts, -90)
        else:
            x_pts, y_pts = rotate(x_pts, y_pts, 90)

        
        return x_pts, y_pts

    
def remove_backtracking_points(x_arr, y_arr, n_iters=10, diff_threshold=1e-15, name=""):
    init_length = len(x_arr)
    length = init_length
    for n in range(n_iters):
        x_pts, y_pts = [], []
        for i, (x1, y1, x2, y2) in enumerate(zip(x_arr[:-1], y_arr[:-1], x_arr[1:], y_arr[1:])):
            if i==0:
                x_pts.append(x1)
                y_pts.append(y1)
            if x2<x1:
                continue
            if (
                (abs(x2-x1) < diff_threshold) and (abs(y2-y1) < diff_threshold)
            ):
                continue
            diff = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if diff < diff_threshold:
                continue
            x_pts.append(x2)
            y_pts.append(y2)
        x_pts = np.asarray(x_pts)
        y_pts = np.asarray(y_pts)
        length_new = len(x_pts)
        if length==length_new:
            print(f"{name} Backtracking: Cleaned in {n+1} iters. Removed {init_length-length} points")
            break
        else:
            length = length_new
            x_arr = x_pts
            y_arr = y_pts
            
    return x_pts, y_pts

def one_stage_filter(
    origin_x, 
    origin_y, 
    n, 
    L, 
    r, 
    l, 
    w, 
    s, 
    x_length, 
    shunt_padding=3, 
    n_points_meander=3600,
    clean=True
):
    Series = BendedEndResonator()
    (x_series, y_series, insert_x, insert_y) = Series.generate_resonator(
        origin_x = origin_x, 
        origin_y = origin_y, 
        N = 2, 
        L = L, 
        r = r, 
        l = l, 
        x_length = x_length,
        upwards = False,
        n_points_meander = 3600,
        clean=clean
    )
    Wreal = (max(x_series)-min(x_series))+(w+2*s)

    # shunt resonator
    above = True if n%2==1 else False
    
    Shunt = ShuntResonator()
    x_shunt1, y_shunt1 = Shunt.generate_resonator(
        origin_x = insert_x[0],
        origin_y = insert_y[0],
        W = Wreal, 
        w = w, 
        s = s, 
        r = r, 
        L = L, 
        x_length = l/2+shunt_padding*r, 
        leftward = False,
        above = above,
        n_points_meander = 3600,
        clean=clean
    )
    return x_series, y_series, x_shunt1, y_shunt1


def two_stage_filter(
    origin_x, 
    origin_y, 
    n, 
    L, 
    r, 
    l, 
    w, 
    s, 
    x_length, 
    shunt_padding=3, 
    n_points_meander=3600,
    clean=True
):
    if n%2 == 1:
        raise ValueError("n must be even")
    # Series Sections
    Series = BendedEndResonator()
    (x_series, y_series, insert_x, insert_y) = Series.generate_resonator(
        origin_x = origin_x, 
        origin_y = origin_y, 
        N = 3, 
        L = L, 
        r = r, 
        l = l, 
        x_length = x_length,
        upwards = False,
        n_points_meander = n_points_meander,
        clean=clean
    )
    Wreal = (max(x_series)-min(x_series))+(w+2*s)

    # Shunted Sections
    Shunt = ShuntResonator()
    x_shunt1, y_shunt1 = Shunt.generate_resonator(
        origin_x = insert_x[0],
        origin_y = insert_y[0],
        W = Wreal, 
        w = w, 
        s = s, 
        r = r, 
        L = L, 
        x_length = l/2+3*r, 
        leftward = False,
        above = False,
        n_points_meander = n_points_meander,
        clean=clean
    )
    x_shunt2, y_shunt2 = Shunt.generate_resonator(
        origin_x = insert_x[1],
        origin_y = insert_y[1],
        W = Wreal, 
        w = w, 
        s = s, 
        r = r, 
        L = L, 
        x_length = l/2+shunt_padding*r, 
        leftward = False,
        above = True,
        n_points_meander = n_points_meander,
        clean=clean
    )
    
    return x_series, y_series, x_shunt1, y_shunt1, x_shunt2, y_shunt2