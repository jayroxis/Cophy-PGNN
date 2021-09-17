import pandas as pd
import numpy as np
from tqdm import tqdm

def solve(solvers, m):
    rt, err, predicted_vecs, predicted_vals = {}, {}, {}, {}
        
    for solver in solvers:
        t, n, e, prediction = solver.run(m, sort=True)
        rt.update(t)
        err.update(e)
        for i in prediction:
            predicted_vecs[i] = [prediction[i][1]]
            predicted_vals[i] = [prediction[i][0]]

    return rt, n, err, predicted_vecs, predicted_vals

def eval_all(gen, solvers):
    g = gen.get()
    t, n, e, predicted_vecs, predicted_vals = solve(solvers, g)
    rt = pd.Series(t)
    err= pd.Series(e)
    norms2= []
    num_runs = gen.count()
    
    predicted_vecs_all = {}
    predicted_vals_all = {}
    for i in predicted_vecs:
        predicted_vecs_all[i] = [predicted_vecs[i]]
    for i in predicted_vals:
        predicted_vals_all[i] = [predicted_vals[i]]
    
    for i in tqdm(range(num_runs-1)):
        t, n, e, predicted_vecs, predicted_vals = solve(solvers, gen.get())
        rt += pd.Series(t)
        err += pd.Series(e)
        norms2.append(n)
        
        for i in predicted_vecs:
            predicted_vecs_all[i].append(predicted_vecs[i])
            
        for i in predicted_vals:
            predicted_vals_all[i].append(predicted_vals[i])
            
            
    for i in predicted_vecs_all:
        predicted_vecs_all[i] = np.concatenate(predicted_vecs_all[i], axis=0)
    
    for i in predicted_vals_all:
        predicted_vals_all[i] = np.concatenate(predicted_vals_all[i], axis=0)
    
    
        
    return pd.DataFrame({'avg time': rt / num_runs, 'avg eigen error': err / num_runs}), norms2, predicted_vecs_all, predicted_vals_all
