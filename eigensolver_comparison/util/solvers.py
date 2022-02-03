
# support libraries
import time
import torch
import numpy as np

# linear algebra libraries
import scipy.linalg
import scipy.sparse.linalg


# ---------------------- Base Class --------------------------
class EigenSolver(object):
    def __init__(self, smallest_eigen=True):
        self.solvers = {}
        if smallest_eigen:
            self.index_of_solution = 0
        else:
            self.index_of_solution = -1
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
            start = time.perf_counter()             # start
            result = solver(m)
            elapse += time.perf_counter() - start   # end
            
            # post-exec sort results
            val, vec = result
            if sort:
                index = np.argsort(val)
                val = val[index]
                vec = vec[:, index]
            
            # record
            runtime[name] = elapse
            val = val[self.index_of_solution]
            vec = vec[:, self.index_of_solution]
            results[name] = (val, vec)
            errors[name] = np.linalg.norm(m @ vec - val * vec)/np.linalg.norm(m @ vec)
            n=np.linalg.norm(vec)
        
        return runtime, n, errors, results
    

# ======================= Numpy ========================
class NumpySolvers(EigenSolver):
    def __init__(self, smallest_eigen=True):
        super().__init__(smallest_eigen)
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
            start = time.perf_counter()             # start
            result = solver(m)
            elapse += time.perf_counter() - start   # end
            
            # post-exec sort results
            val, vec = result
            if sort:
                index = np.argsort(val)
                val = val[index]
                vec = vec[:, index]
            
            # record
            val = val[self.index_of_solution]
            vec = vec[:, self.index_of_solution]
            runtime[name] = elapse
            results[name] = (val, vec)
            errors[name] = np.linalg.norm(m @ vec - val * vec)/np.linalg.norm(m @ vec)
            n=np.linalg.norm(vec)
            
        return runtime, n, errors, results
    
    
# ======================= Scipy ========================
class ScipySolvers(EigenSolver):
    def __init__(self, smallest_eigen=True):
        super().__init__(smallest_eigen)
        self.solvers = {
            'scipy_eig': scipy.linalg.eig,
        }
          

class ScipySparse(EigenSolver):
    def __init__(self, smallest_eigen=True):
        super().__init__(smallest_eigen)
        self.solvers = {
            'scipy_sparse_eig': lambda m: scipy.sparse.linalg.eigs(m, k=m.shape[1]),
        }
            
            
# ======================= PyTorch ========================      
# TODO: This function is depricated and is not working properly for complex eigenvectors. Needs to be replaced with torch.linalg.eig()      
class TorchSolvers(EigenSolver):
    def __init__(self, smallest_eigen=True):
        super().__init__(smallest_eigen)
         
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
            start = time.perf_counter()             # start
            with torch.no_grad():
                result = solver(m)
            elapse += time.perf_counter() - start   # end
            
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
            if val[-1, 0] != val[-2, 0]: # real
                val_last = val[self.index_of_solution, 0]
                vec_last = vec[:,self.index_of_solution]
            else: # complex conjugate
                if self.index_of_solution == -1:
                    self.index_of_solution = -2
                val_last = val[self.index_of_solution, 0] + 1j*val[self.index_of_solution, 1] 
                vec_last = vec[:,self.index_of_solution]+1j*vec[:,self.index_of_solution+1]
            errors[name] = np.linalg.norm(m @ vec_last - val_last * vec_last)/np.linalg.norm(m @ vec_last)
            n=np.linalg.norm(vec_last)
            results[name] = (val_last, vec_last)
            
        return runtime, n, errors, results