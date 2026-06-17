from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from casadi import *

from src import mocca, unicircle

def main():
    l_veh = 5.0
    w_veh = 2.2
    sigma_vec = [0.5, 0.5, 0.5] # standard deviation [x, y, theta] (not the covariance matrix!!!)
    
    f_mocca = mocca()
    f_unicircle = unicircle()
    n_funcs = 2
    
    ## Scenario B
    n_testcases = 3    
    p_colls = np.zeros((n_funcs, n_testcases, 151))
    
    plt.subplots(n_testcases, 1)
    for itc, tc in (pbar_tc:=tqdm(enumerate(range(n_testcases)), total=n_testcases, colour="red", leave=False)):
        pbar_tc.set_description(f"SCENARIO B: testcase {itc+1}")
        if tc==0:
            xs = np.ones((151,))*(l_veh/2+w_veh)
            ys = np.linspace(-7.5, 7.5, 151)
        elif tc==1:
            xs = np.zeros((151,))
            ys = np.linspace(-7.5, 7.5, 151)
        elif tc==2:
            xs = np.linspace(-7.5, 7.5, 151)
            ys = np.linspace(-7.5, 7.5, 151)
        else:
            continue
        for ixy, [x,y] in (pbar_xy:=tqdm(enumerate(zip(xs, ys)), total=xs.shape[0], colour="blue", leave=False)):
            pbar_xy.set_description(f"SCENARIO B: y-distance={y:.2f}")
            
            ## RUN MoCCA
            p_coll_mocca = f_mocca(
                DM.zeros(6, 1),
                vertcat(DM([x, y, np.deg2rad(90)]), DM.zeros(3, 1), DM(l_veh), DM(w_veh)),
                DM(w_veh),
                DM(l_veh),
                DM(0.0),
                DM(sigma_vec)
            )
            p_colls[0, itc, ixy] = p_coll_mocca[0]
            
            ## RUN unicircle Method
            p_coll_unicircle = f_unicircle(
                DM([0]),
                vertcat(DM([x, y]), DM([*sigma_vec[:2], np.sqrt((l_veh/2)**2 + (w_veh/2)**2)*2]))
            )
            p_colls[1, itc, ixy] = p_coll_unicircle[0]
        ## PLOT results  
        plt.subplot(n_testcases, 1, itc+1)
        plt.cla()
        for nf in range(n_funcs):
            plt.plot(ys, p_colls[nf, itc, :])
        if(itc==0):
            plt.title("Scenario B")

    
if __name__=="__main__":
    main()
    plt.show()