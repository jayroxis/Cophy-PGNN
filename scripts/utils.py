import os, shutil
import glob

def free_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
                
def free_logs(log_path='//home/jayroxis/Condensed Matter Theory/logs/'):
    free_dir(log_path)

def free_results(results_path='//home/jayroxis/Condensed Matter Theory/results/'):
    free_dir(results_path)

def free_models(model_path='//home/jayroxis/Condensed Matter Theory/models/'):
    free_dir(model_path)
    
def free_all(base_dir='//home/jayroxis/Condensed Matter Theory/'):
    free_dir(base_dir + 'logs/')
    free_dir(base_dir + 'models/')
    free_dir(base_dir + 'results/')
    
    num_logs = len([name for name in os.listdir(base_dir + 'logs/') if os.path.isfile(name)])
    num_models = len([name for name in os.listdir(base_dir + 'models/') if os.path.isfile(name)])
    num_results = len([name for name in os.listdir(base_dir + 'results/') if os.path.isfile(name)])
    
    if num_logs == 0 and num_results == 0 and num_models == 0:
        print('Successful.')
        return True
    else:
        print('Attempt failed.')
        return False
    
