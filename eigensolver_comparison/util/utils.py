import pandas as pd
import numpy as np
from tqdm import tqdm

def solve(solvers, m):
    rt, err = {}, {}
    for solver in solvers:
        t, n, e = solver.run(m, sort=False)
        rt.update(t)
        err.update(e)
    return rt, n, err

def eval_all(gen, solvers):
    g = gen.get()
    t, n, e = solve(solvers, g)
    rt = pd.Series(t)
    err= pd.Series(e)
    norms2= []
    num_runs = gen.count()
    for i in tqdm(range(num_runs-1)):
        t, n, e = solve(solvers, gen.get())
        rt += pd.Series(t)
        err += pd.Series(e)
        norms2.append(n)
        
    return pd.DataFrame({'avg time': rt / num_runs, 'avg L1 error': err / num_runs}), norms2
