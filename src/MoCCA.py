import os

from casadi import *
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors

def getDistanceFunction() -> Function:
    """calculates the distance "dist" between two points on "g(t)" and "h(s)"
    
    g(t) = A + t(B-A)
    h(s) = C + s(D-C)
    
    Parameters
    -----------
    A: SX
        point A = [xa, ya]
    B: SX
        point B = [xb, yb]
    C: SX
        point C = [xc, yc]
    D: SX
        point D = [xd, yd]
    s: SX
        scalar s 

    Returns:
        casadi.Function:
            a function "distance_func" [A, B, C, D, s, t] -> [dist]
    """
    
    A = SX.sym("A", 2, 1)
    B = SX.sym("B", 2, 1)
    C = SX.sym("C", 2, 1)
    D = SX.sym("D", 2, 1)
    t = SX.sym("t", 1, 1)
    s = SX.sym("s", 1, 1)
    # extract coordinates
    xa = A[0]
    ya = A[1]
    xb = B[0]
    yb = B[1]
    xc = C[0]
    yc = C[1]
    xd = D[0]
    yd = D[1]
    
    # calculate important differences
    xca = xc-xa
    yca = yc-ya
    xdc = xd-xc
    ydc = yd-yc
    xba = xb-xa
    yba = yb-ya
    
    ## non Parallel
    c1 = xca*xca + yca*yca
    c2 = 2 * (xca*xdc + yca*ydc)
    c3 = 2 * (xca*xba + yca*yba)
    c4 = 2 * (xdc*xba + ydc*yba)
    c5 = xdc*xdc + ydc*ydc
    c6 = xba*xba + yba*yba
    
    dist = c1 + s*c2 - t*c3 - s*t*c4 +s*s*c5 + t*t*c6
    return Function("distance_func", [A, B, C, D, s, t], [dist], ["A", "B", "C", "D", "s", "t"], ["dist"], dict(cse=True))


def findNearestPoints() -> Function:
    """find the 2 nearest points on the lines AB and CD
    
    g(t) = A + t(B-A)
    h(s) = C + s(D-C)
    
    minimizing the distance between g & s
    dist² = || h-g ||²
    
    search the minimum of
    
    d dist²        
    --------  = c2 + 2*s*c5 - t*c4
    d s
    
    and 
    
    d dist²       
    -------- = -c3 + 2*t*c6 - s*c4
    d t
    
    solving for all 2 possible point-pairs, and therefor 
    ss,min <-> t1,min
    ts,min <-> s1,min
    
    
    Parameters
    -----------
    A: SX
        point A = [xa, ya]
    B: SX
        point B = [xb, yb]
    C: SX
        point C = [xc, yc]
    D: SX
        point D = [xd, yd]
    
    Returns
    -------
    casadi.Function:
        a Function f calculating the nearest Points of two limited lines
        f([A, B, C, D]) -> [E, F]
    """
    A = SX.sym("A", 2, 1)
    B = SX.sym("B", 2, 1)
    C = SX.sym("C", 2, 1)
    D = SX.sym("D", 2, 1)
    # extract coordinates
    xa = A[0]
    ya = A[1]
    xb = B[0]
    yb = B[1]
    xc = C[0]
    yc = C[1]
    xd = D[0]
    yd = D[1]
    
    # calculate important differences
    xca = xc-xa
    yca = yc-ya
    xdc = xd-xc
    ydc = yd-yc
    xba = xb-xa
    yba = yb-ya
    xda = xd-xa
    yda = yd-ya
    xbc = xb-xc
    ybc = yb-yc
    
    ## non Parallel
    c1 = xca*xca + yca*yca
    c2 = 2 * (xca*xdc + yca*ydc)
    c3 = 2 * (xca*xba + yca*yba)
    c4 = 2 * (xdc*xba + ydc*yba)
    c5 = xdc*xdc + ydc*ydc
    c6 = xba*xba + yba*yba
    
    f_dist = getDistanceFunction()
    
    s1 = (c3*c4 - 2*c2*c6)/(4*c5*c6 - c4*c4)
    t2 = (2*c3*c5 - c2*c4)/(4*c5*c6 - c4*c4)
    
    s1min = fmin(fmax(s1, 0), 1)  # limit between 0 and 1
    t2min = fmin(fmax(t2, 0), 1)  # limit between 0 and 1
    
    t1 = (c2 + 2*s1min*c5)/c4
    s2 = (t2min*c4 - c2)/(2*c5)
    
    t1min = fmin(fmax(t1, 0), 1)  # limit between 0 and 1
    s2min = fmin(fmax(s2, 0), 1)  # limit between 0 and 1
    
    dist_s2t1 = f_dist(A, B, C, D, s1min, t1min)[0]
    dist_t2s1 = f_dist(A, B, C, D, s2min, t2min)[0]
    Enp = if_else(dist_s2t1 < dist_t2s1, A + t1min*(B-A), A + t2min*(B-A))
    Fnp = if_else(dist_s2t1 < dist_t2s1, C + s1min*(D-C), C + s2min*(D-C))
    
    ## Parallel
    # checking distances for each point
    sa = -(xdc*xca + ydc*yca)/(xdc*xdc + ydc*ydc)
    sb = (xdc*xbc + ydc*ybc)/(xdc*xdc + ydc*ydc)
    tc = (xba*xca + yba*yca)/(xba*xba + yba*yba)
    td = (xba*xda + yba*yda)/(xba*xba + yba*yba)
    
    samin = fmin(fmax(sa, 0), 1)
    sbmin = fmin(fmax(sb, 0), 1)
    tcmin = fmin(fmax(tc, 0), 1)
    tdmin = fmin(fmax(td, 0), 1)
    
    LA = C + samin*(D-C)
    LB = C + sbmin*(D-C)
    LC = A + tcmin*(B-A)
    LD = A + tdmin*(B-A)
    
    dists = SX(4,1)
    dists[0] = f_dist(A, B, C, D, samin, SX.zeros(1,1))
    dists[1] = f_dist(A, B, C, D, sbmin, SX.ones(1,1))
    dists[2] = f_dist(A, B, C, D, SX.zeros(1,1), tcmin)
    dists[3] = f_dist(A, B, C, D, SX.ones(1,1), tdmin)
    
    # tolerance between changing points
    # otherwise due to numerical inaccuracies the points may jump randomly
    min_dist_arr = if_else(fabs(mmin(dists)-dists[0])<1e-3, SX([1, 0, 0, 0]),
                   if_else(fabs(mmin(dists)-dists[1])<1e-3, SX([0, 1, 0, 0]),
                   if_else(fabs(mmin(dists)-dists[2])<1e-3, SX([0, 0, 1, 0]), 
                   if_else(fabs(mmin(dists)-dists[3])<1e-3, SX([0, 0, 0, 1]),
                                                            SX([0, 0, 0, 1]))))) 

    Ep = SX(2, 1)
    Fp = SX(2, 1)
    Ep[0] = dot(min_dist_arr, vertcat(A[0], B[0], LC[0], LD[0]))
    Ep[1] = dot(min_dist_arr, vertcat(A[1], B[1], LC[1], LD[1]))
    Fp[0] = dot(min_dist_arr, vertcat(LA[0], LB[0], C[0], D[0]))
    Fp[1] = dot(min_dist_arr, vertcat(LA[1], LB[1], C[1], D[1]))
    
    ## check if parallel or not
    E = if_else(fabs(dot((B-A)/norm_2(B-A), (D-C)/norm_2(D-C))) < 1, Enp, Ep)
    F = if_else(fabs(dot((B-A)/norm_2(B-A), (D-C)/norm_2(D-C))) < 1, Fnp, Fp)
    return Function("f_nearest_points", [A, B, C, D], [E, F, dot((B-A)/norm_2(B-A), (D-C)/norm_2(D-C)), dists, min_dist_arr], ["A", "B", "C", "D"], ["E", "F", "dot", "dists_sel", "min_dists"], dict(cse=True))


def updateVariance() -> Function:
    """updates the covariance matrix sigma by propagating the rotational uncertainty
    
    sigma = [[sigma_xx**2, sigma_xy**2], [sigma_yx**2, sigma_yy**2]]
    J = [[1 0 -sin(µtheta)*sigma_theta**2],
         [0 1  cos(µtheta)*sigma_theta**2]]
    sigma = J**T @ sigma @ J

    Parameters
    -----------
    Xobj: SX
        State of the Object [µx, µy, µtheta, sigma_x, sigma_y, sigma_theta, width, length]
    F: SX
        Point F = [xf, yf]
    
    Returns
    -------
        casadi.Function: 
            the function "sigma_update" propagating the covariance matrix ot point F
            [Xobj, F] -> [sigma]
    """
    Xobj = SX.sym("Xobj", 8, 1) # µx, µy, µtheta, sigma_x, sigma_y, sigma_theta, width, length
    F = SX.sym("F", 2, 1)
    
    OF = Xobj[:2] - F
    l = norm_2(OF)
    
    sigma = SX(4, 1)    
    sigma[0] = Xobj[3]**2 + (-l*sin(Xobj[2]) * Xobj[5])**2
    sigma[1] = -l**2*sin(Xobj[2])*cos(Xobj[2])*Xobj[5]**2
    sigma[2] = sigma[1]
    sigma[3] = Xobj[4]**2 + ( l*cos(Xobj[2]) * Xobj[5])**2

    return Function("sigma_update", [Xobj, F], [sigma], ["Xobj", "F"], ["sigma"], dict(cse=True))


def changeReferenceCOS() -> Function:
    """translates point F to the Coordinate System on E
    
    Parameters
    -----------
    E: SX
        point E = [xe, ye]
    F: SX
        point F = [xf, yf]

    Returns
    -------
        casadi.Funtion: 
            a function "changeRefPoints" translating point F into local coordinate system at E
            [E, F] -> [SX([0,0]), F-E]
    """
    F = SX.sym("F", 2, 1)
    E = SX.sym("E", 2, 1)
    return Function("changeRefPoints", [E,F], [SX.zeros(2,1), F-E])


def simpson(f:Function, lb:float, ub:float, params:SX) -> SX:
    """Applies Simpson's one-third-rule of function f between lb and ub

    Parameters
    -----------
        f: (casadi.Function) 
            the used function to integrate
        lb: float
            lower bound of integration intervall
        ub: float
            upper bound of integration intervall 
        params: SX
            parameters needed for function f besides x

    Returns
    -------
        SX:
            simpson integral of function f between lb and ub
    """
    midpoint = (lb+ub)/2.0
    return (ub-lb)/6.0 * (f(lb,params) + 4*f(midpoint,params) + f(ub,params))


def createCollisionProbability(substeps:int=80) -> Function:
    """Calculates the circular integral of a 2D probability density function with uncorrelated variables.

    Parameters
    -----------
        substeps (int, optional): number of substeps. Defaults to 80.
    

    Returns
    -------
        Function: function "calc_circ" to calculate the integral over the circular Area of radius R_hbr.
        [y0, [µx, µy, sigma_x, sigma_y, R_hbr]] -> [coll_prob]
    
    ARGS:
    -----
        y0: SX
            Start value to integrate to
        µx: SX 
            mean x-position of probability distribution
        µy: SX
            mean y-positino of probability distribution
        sigma_x: SX
            standard deviation (not variance!!) in x-direction
        sigma_y: SX
            standard deviation (not variance!!) in y-direction
        R_hbr: SX
            hard body radius to integrate over
    """
    mu_x = SX.sym("mu_x")
    mu_y = SX.sym("mu_y")
    sigma_x = SX.sym("sigma_x", 1, 1)
    sigma_y = SX.sym("sigma_y", 1, 1)
    R_hbr = SX.sym("R_hbr", 1, 1)
    # define symbolic ode variables
    y0 = SX.sym("y0")
    x = SX.sym("x")
    
    p = vertcat(mu_x, mu_y, sigma_x, sigma_y, R_hbr)

    ode =   exp(-(x-mu_x)**2/(2*sigma_x**2)) * (
                erf((sqrt(R_hbr**2-x**2)-mu_y)/(sqrt(2)*sigma_y)) +  # lower bound
                erf((sqrt(R_hbr**2-x**2)+mu_y)/(sqrt(2)*sigma_y))    # upper bound
            )
    y_sum = y0
    f_ode = Function("ode",[x,p],[ode], dict(cse=True))
    integration_steps = substeps//5
    for i in range(1,integration_steps+1):
        lb = -R_hbr + (i-1)/integration_steps*(2*R_hbr)
        ub = -R_hbr + i/integration_steps*(2*R_hbr)
        mb = (ub+lb)/2.0
        S_ab = simpson(f_ode, lb, ub, p)
        S_ac = simpson(f_ode, lb, mb, p)
        S_cb = simpson(f_ode, mb, ub, p)
        S_2 = S_ac + S_cb
        y_sum += S_2 + (S_2-S_ab)/15.0
    y_sum *= 1/(2*sqrt(2*np.pi)*sigma_x)
    coll_prob = Function("calc_circ", [y0, p], [y_sum], ["x0", "p"], ["xf"], dict(cse=True))
    return coll_prob


def createCollisionProbability_decorrelated(substeps:int=80) -> Function:
    """Calculates the circular integral of a 2D probability density function with correlated variables.
    By decorrelating the distribution using Principal-axis decomposition.

    Parameters
    -----------
        substeps (int, optional): number of substeps. Defaults to 80.
    

    Returns
    -------
        Function: function "calc_circ" to calculate the integral over the circular Area of radius R_hbr.
        [y0, [µx, µy, sigma_x, sigma_y, R_hbr]] -> [coll_prob]
    
    ARGS:
    -----
        y0: SX
            Start value to integrate to
        µx: SX 
            mean x-position of probability distribution
        µy: SX
            mean y-positino of probability distribution
        sigma_x: SX
            standard deviation (not variance!!) in x-direction
        sigma_y: SX
            standard deviation (not variance!!) in y-direction
        R_hbr: SX
            hard body radius to integrate over
    """
    mu_x = SX.sym("mu_x")
    mu_y = SX.sym("mu_y")
    sx_sigma = SX.sym("sigma", 2, 2)
    sx_sigma_decorrelated = SX(2, 1)
    R_hbr = SX.sym("R_hbr")
    y0 = SX.sym("y0")
    
    p_cov = vertcat(mu_x, mu_y, sx_sigma[0,0], sx_sigma[0,1], sx_sigma[1,0], sx_sigma[1,1], R_hbr)
    
    # calc eigen vector:
    eig_vals = eig_symbolic(sx_sigma)
    eig_vecs = SX(sx_sigma.size1(), sx_sigma.size2())
    for i in range(eig_vals.size1()):
         tmp_v1 = vertcat(sx_sigma[0,1], eig_vals[i]-sx_sigma[0,0])
         tmp_v2 = vertcat(eig_vals[i]-sx_sigma[1,1], sx_sigma[1,0])
         tmp_v = if_else(norm_2(tmp_v1)<1e-6, tmp_v2, tmp_v1)
         eig_vecs[i,:] = tmp_v /norm_2(tmp_v)
    # calculate rotation angle
    Theta = -arctan2(eig_vecs[0,1], eig_vecs[0, 0])
    rot_mat = horzcat(vertcat(cos(Theta),-sin(Theta)),vertcat(sin(Theta),cos(Theta)))
    mu = horzcat(mu_x, mu_y)
    mu_decorrelated = mu @ rot_mat
    sx_sigma_decorrelated = eig_vals
    
    p_orig = vertcat(mu_x, mu_y, sqrt(sx_sigma[0,0]), sqrt(sx_sigma[1,1]), R_hbr)
    p_decorrelated = vertcat(mu_decorrelated[0], mu_decorrelated[1], sqrt(sx_sigma_decorrelated[0]), sqrt(sx_sigma_decorrelated[1]), R_hbr)
    p = if_else(fabs(fabs(sx_sigma[0,1]))<1e-4, p_orig, p_decorrelated)
    f_coll_prob = createCollisionProbability(substeps)
    xf = f_coll_prob(y0, p)
    coll_prob = Function("collProb_decorrelated", [y0, p_cov], [xf], ["x0", "p"], ["xf"], dict(cse=True))
    return coll_prob


def createCollProbFun(substeps=80) -> Function:
    """main function of this file. creating the whole structure for the MoCCA Algorithm.
    1. find the nearest points on the centerline of each vehicle,
    2. update the variance according to the displacement,
    3. update the reference coordinate system with point E as origin,
    4. decorrelate the random variables,
    5. integrate over the circle.
    

    Parameters
    -----------
        substeps (int, optional): _description_. Defaults to 80.

    Returns
    ---------
        Function: complete Function to execute MoCCA.
        [Xego, Xobj, we, le, ds, [sigma_x, sigma_y, sigma_theta]] -> [cost, E, F] 
        
    ARGS
    -----
        Xego:           SX
                        state of the ego vehicle [x, y, theta, v, \dot{theta}, a]
        Xobj:           SX
                        state of the object vehicle [x, y, theta, v, \dot{theta}, y, length, width]
        we:             SX  
                        width of the ego vehicle
        le:             SX
                        length of the ego vehicle
        ds:             SX
                        additional safety distance to accomodate for the underapproximation
        sigma_x:        SX
                        standard deviation (not variance!!!) in x direction of the detected object
        sigma_y:        SX
                        standard deviation (not variance!!!) in y direction of the detected object
        sigma_theta:    SX
                        standard deviation (not variance!!!) of the orientation of the detected object
    """
    f_nearestPoints = findNearestPoints()
    f_sigmaUpdate = updateVariance()
    f_updateCOS = changeReferenceCOS()
    f_collProb = createCollisionProbability_decorrelated(substeps)
    
    sx_xego = SX.sym("Xego", 6, 1)
    sx_we = SX.sym("we", 1, 1)
    sx_le = SX.sym("le", 1, 1)
    Xego = vertcat(sx_xego, sx_le, sx_we)
    sx_xobj = SX.sym("Xobj", 8, 1)
    sx_sigma = SX.sym("sigma", 3, 1) 
    ds = SX.sym("rsafe", 1, 1)
    Xobj = vertcat(sx_xobj[:3,0], sx_sigma, sx_xobj[-2:,0])
    R_hbr = sqrt(Xobj[7]**2/2) + sqrt(Xego[7]**2/2) + ds
    
    A = SX(2,1)
    B = SX(2,1)
    C = SX(2,1)
    D = SX(2,1)
    
    # determine the points for the ego vehicle
    A[0] = Xego[0] - (Xego[6]-Xego[7])/2 * cos(Xego[2])
    A[1] = Xego[1] - (Xego[6]-Xego[7])/2 * sin(Xego[2])
    B[0] = Xego[0] + (Xego[6]-Xego[7])/2 * cos(Xego[2])
    B[1] = Xego[1] + (Xego[6]-Xego[7])/2 * sin(Xego[2])
    
    # determine the points for the object vehicle
    C[0] = Xobj[0,0] - (Xobj[6]-Xobj[7])/2 * cos(Xobj[2])
    C[1] = Xobj[1,0] - (Xobj[6]-Xobj[7])/2 * sin(Xobj[2])
    D[0] = Xobj[0,0] + (Xobj[6]-Xobj[7])/2 * cos(Xobj[2])
    D[1] = Xobj[1,0] + (Xobj[6]-Xobj[7])/2 * sin(Xobj[2])
    
    nearest_points = f_nearestPoints(A, B, C, D)
    E = nearest_points[0]
    F = nearest_points[1]
    new_sigma = f_sigmaUpdate(Xobj[:,0], F)
    newPoints = f_updateCOS(E,F)
    cost = f_collProb(SX(0), vertcat(newPoints[1], new_sigma, R_hbr))
    
    model = Function(f"calc_var_circ_{substeps}substeps", 
                     [sx_xego, sx_xobj, sx_we, sx_le, ds, sx_sigma], 
                     [cost, E, F], 
                     ["state", "objects", "mocca.w_e", "mocca.l_e", "mocca.r_safe", "mocca.sigma"], 
                     ["cost", "mocca.E", "mocca.F"],
                     {"cse": True})
    
    return model


### MARK:
if __name__=="__main__":    
    model = createCollProbFun()
    if model:
        file_name = os.path.basename(__file__).split(".")[0]+".casadi"
        save_path = os.path.join(os.path.dirname(__file__),file_name)
        model.save(save_path)
        print(model)
