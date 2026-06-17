import time
from tqdm import tqdm
import numpy as np
import pandas as pd
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
    
    ## Scenario A
    xs = np.linspace(7.5, -7.5, 151)
    ys = np.array([0, w_veh/2, 3/2*w_veh])
    
    p_colls = np.zeros((n_funcs, ys.shape[0], xs.shape[0]))
    
    plt.subplots(ys.shape[0], 1)
    for iy, y in (pbar_y:=tqdm(enumerate(ys), total=ys.shape[0], colour="red", leave=False)):
        pbar_y.set_description(f"SCENARIO A: testcase {iy+1}")
        for ix, x in (pbar_x:=tqdm(enumerate(xs), total=xs.shape[0], colour="blue", leave=False)):
            pbar_x.set_description(f"SCENARIO A: x-distance={x:.2f}")
            
            p_coll_mocca = f_mocca(
                DM.zeros(6, 1),
                vertcat(DM([x, y, 0]), DM.zeros(3, 1), DM(l_veh), DM(w_veh)),
                DM(w_veh),
                DM(l_veh),
                DM(0.0),
                DM(sigma_vec)
            )
            p_colls[0, iy, ix] = p_coll_mocca[0]
            
            p_coll_unicircle = f_unicircle(
                DM([0]),
                vertcat(DM([x, y]), DM([*sigma_vec[:2], np.sqrt((l_veh/2)**2 + (w_veh/2)**2)*2]))
            )
            p_colls[1, iy, ix] = p_coll_unicircle[0]
            
        plt.subplot(ys.shape[0], 1, iy+1)
        plt.cla()
        for nf in range(n_funcs):
            plt.plot(xs, p_colls[nf, iy, :])
        if(iy==0):
            plt.title("Scenario A")
    
if __name__=="__main__":
    main()
    plt.show()
    