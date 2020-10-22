
# support libraries
import time
import torch
import numpy as np

# linear algebra libraries
import scipy.linalg
import scipy.sparse.linalg


# ---------------------- Base Class --------------------------
class EigenSolver(object):
    def __init__(self):
        self.solvers = {}
        pass
    
    def init_dict(self):
        d = dict(self.solvers)
        for key in d:
            d[key] = 0
        return d
    
    def run(self, m, sort=True):
        m = m[1]
        if torch.is_tensor(m):
            m.detach().cpu().numpy()
            
        results = self.init_dict()
        runtime = self.init_dict()
        errors = self.init_dict()
        
        for name in self.solvers:
            solver = self.solvers[name]
            elapse = 0
            
            # solve and record runtime
            start = time.process_time()             # start
            result = solver(m)
            elapse += time.process_time() - start   # end
            
            # post-exec sort results
            val, vec = result
            if sort:
                index = np.argsort(val)
                val = val[index]
                vec = vec[:, index]
            
            # record
            runtime[name] = elapse
            results[name] = (val, vec)
            errors[name] = np.mean(np.abs(m @ vec[:, -1] - val[-1] * vec[:, -1]))
            n=np.linalg.norm(vec[:, -1])
            
        return runtime, n, errors
    

# ======================= Numpy ========================
class NumpySolvers(EigenSolver):
    def __init__(self):
        super().__init__()
        self.solvers = {
            'numpy_eig': np.linalg.eig,
        }
        
    def run(self, m, sort=True):
        m = m[1]
        if torch.is_tensor(m):
            m.detach().cpu().numpy()
        
        results = self.init_dict()
        runtime = self.init_dict()
        errors = self.init_dict()
        
        for name in self.solvers:
            solver = self.solvers[name]
            elapse = 0
            
            # solve and record runtime
            start = time.process_time()             # start
            result = solver(m)
            elapse += time.process_time() - start   # end
            
            # post-exec sort results
            val, vec = result
            if sort:
                index = np.argsort(val)
                val = val[index]
                vec = vec[:, index]
            
            # record
            runtime[name] = elapse
            results[name] = (val, vec)
            errors[name] = np.mean(np.abs(m @ vec[:, -1] - val[-1] * vec[:, -1]))
            n=np.linalg.norm(vec[:, -1])
            
        return runtime, n, errors
    
    
# ======================= Scipy ========================
class ScipySolvers(EigenSolver):
    def __init__(self):
        super().__init__()
        self.solvers = {
            'scipy_eig': scipy.linalg.eig,
        }
          

class ScipySparse(EigenSolver):
    def __init__(self):
        super().__init__()
        self.solvers = {
            'scipy_sparse_eig': scipy.sparse.linalg.eigs,
        }
            
            
# ======================= PyTorch ========================            
class TorchSolvers(EigenSolver):
    def __init__(self):
        super().__init__()
         
        self.solvers = {
            'torch_cuda': lambda m: torch.eig(m, eigenvectors=True),
        }
    
    def run(self, m, sort=True):
        m = m[1]
        if torch.is_tensor(m):
            m.detach().cpu().numpy()
            
        results = self.init_dict()
        runtime = self.init_dict()
        errors = self.init_dict()
        
        for name in self.solvers:
            solver = self.solvers[name]
            elapse = 0
            
            if 'cuda' in name:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
                
            m = torch.tensor(m, device=device, dtype=torch.float64)
            
            # solve and record runtime
            start = time.process_time()             # start
            with torch.no_grad():
                result = solver(m)
            elapse += time.process_time() - start   # end
            
            # post-exec sort results
            val, vec = result
            val = val.cpu().numpy()
            vec = vec.cpu().numpy()
            m = m.cpu().numpy()
            
            if sort:
                index = np.argsort(val[:,0])
                val = val[index]
                vec = vec[:, index]
            
            # record
            runtime[name] = elapse
            results[name] = (val, vec)
            if val[-1, 0] != val[-2, 0]: # real
                val_last = val[-1, 0]
                vec_last = vec[:,-1]
            else: # complex conjugate
                val_last = val[-2, 0] + 1j*val[-2, 1]
                vec_last = vec[:,-2]+1j*vec[:,-1]
            errors[name] = np.mean(np.abs(m @ vec_last - val_last * vec_last))
            n=np.linalg.norm(vec_last)
            
        return runtime, n, errors