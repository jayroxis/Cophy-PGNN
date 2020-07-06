import torch
import numpy as np
import os
import pickle
import pandas as pd
import glob

def load_grad(grad_path, epoch):
    load = lambda path: pickle.load(open(grad_path + path % epoch, 'rb'))
    grad_dict = {
        'train_mse': load('train_mse_%d.pkl'),
        'train_s_norm': load('train_train_s_norm_%d.pkl'),
        'test_s_norm': load('test_s_norm_%d.pkl'),
        'train_e': load('train_e_%d.pkl'),
        'test_e': load('test_e_%d.pkl'),
        'train_se': load('train_se_%d.pkl'),
        'test_se': load('test_se_%d.pkl'),
        'train_all': load('train_all_%d.pkl'),
        'test_all': load('test_all_%d.pkl')
    }
    return grad_dict

def cosine(w1, w2):
    numerator = torch.sum(w1 * w2)
    denominator = torch.sqrt(torch.sum(w1 ** 2) * torch.sum(w2 ** 2))
    return float(numerator / denominator)

def optimal_direction(state_path, epoch, final_epoch):
    current_state = torch.load(state_path + 'state_%d.pt' % epoch)
#     final_state = torch.load(state_path + 'state_%d.pt' % final_epoch)
    model_file = glob.glob(state_path.replace('/states/', '/models/')+"*")[0]
    final_state = torch.load(model_file)
    direction = {}
    for key in current_state.keys():
        direction[key] = final_state[key] - current_state[key]
    return direction

def grad_angle(grad_path, state_path, epoch, final_epoch):
    epoch += 1
    final_epoch += 1
    opt = optimal_direction(state_path, epoch, final_epoch)
    grad = load_grad(grad_path, epoch)
    angles = {}
    for key in opt.keys():
        angle = {}
        for loss in grad.keys():
            # update direction is opposite to the direction of grad
            angle[loss] = -cosine(opt[key], grad[loss][key])
        angles[key] = angle
    return angles

def calc_angles(model, grad_angles):
    param_angles = {}
    for p, value in grad_angles[model][0].items():
        param_angles[p] = {}
        for loss in value.keys():
            param_angles[p][loss] = []
    for angle in grad_angles[model]:
        for p, d in angle.items():
            for loss, value in d.items():
                param_angles[p][loss].append(value)
    for p in param_angles.keys():
        for loss in param_angles[p].keys():
            param_angles[p][loss] = np.array(param_angles[p][loss])
        return param_angles


def projection(direction, grad):
    numerator = torch.sum(direction * grad)
    denominator = torch.sqrt(torch.sum(direction ** 2))
    return float(numerator / denominator)
    
def grad_projection(grad_path, state_path, epoch, final_epoch):
    epoch += 1
    final_epoch += 1
    opt = optimal_direction(state_path, epoch, final_epoch)
    grad = load_grad(grad_path, epoch)
    projects = {}
    for key in opt.keys():
        project = {}
        for loss in grad.keys():
            # update direction is opposite to the direction of grad
            project[loss] = - projection(opt[key], grad[loss][key])
        projects[key] = project
    return projects

def calc_projections(model, grad_projections):
   
    param_projections = {}
    for p, value in grad_projections[model][0].items():
        param_projections[p] = {}
        for loss in value.keys():
            param_projections[p][loss] = []
    for project in grad_projections[model]:
        for p, d in project.items():
            for loss, value in d.items():
                param_projections[p][loss].append(value)
    for p in param_projections.keys():
        for loss in param_projections[p].keys():
            param_projections[p][loss] = np.array(param_projections[p][loss])
        return param_projections

def concat_params(params):
    flatten = []
    for value in params.values():
        flatten.append(value.view(-1))
    return torch.cat(flatten, dim=0)

def overall_projection(grad_path, state_path, epoch, final_epoch):
    epoch += 1
    final_epoch += 1
    opt = optimal_direction(state_path, epoch, final_epoch)
    grad = load_grad(grad_path, epoch)
    projects = {}
    for loss in grad.keys():
        projects[loss] = - projection(concat_params(opt), concat_params(grad[loss]))
    return projects

def overall_angle(grad_path, state_path, epoch, final_epoch):
    epoch += 1
    final_epoch += 1
    opt = optimal_direction(state_path, epoch, final_epoch)
    grad = load_grad(grad_path, epoch)
    results = {}
    for loss in grad.keys():
        results[loss] = - cosine(concat_params(opt), concat_params(grad[loss]))
    return results

def calc_overall_projection():
    os.chdir('//home/jayroxis/Condensed Matter Theory/gradients/')
    overall_projections = {}
    for path in glob.glob('*'):
        name = path.split('_')[0]
        print(name, path)
        grad_path = path + '/grads/'
        state_path = path + '/states/'
        
        overall_projections[name] = []
        for epoch in range(499):
            project = overall_projection(grad_path, state_path, epoch, max(range(500)))
            overall_projections[name].append(project)
            
    for model in overall_projections.keys():
        losses = overall_projections[model][0].keys()
        values = [list(item.values()) for item in overall_projections[model]]
        values = np.stack(values)
        df_projects = pd.DataFrame(columns=losses, data=values)
        overall_projections[model] = df_projects
    return overall_projections

def calc_overall_angle():
    os.chdir('//home/jayroxis/Condensed Matter Theory/gradients/')
    overall_angles = {}
    for path in glob.glob('*'):
        name = path.split('_')[0]
        print(name, path)
        grad_path = path + '/grads/'
        state_path = path + '/states/'
        
        overall_angles[name] = []
        for epoch in range(499):
            project = overall_angle(grad_path, state_path, epoch, max(range(500)))
            overall_angles[name].append(project)
            
    for model in overall_angles.keys():
        losses = overall_angles[model][0].keys()
        values = [list(item.values()) for item in overall_angles[model]]
        values = np.stack(values)
        df_angles = pd.DataFrame(columns=losses, data=values)
        overall_angles[model] = df_angles
    return overall_angles