"""Gradient, sensor dependency and causal preflight for visual-only training."""
import torch

def visual_preflight(model,sample,device):
    import copy
    from lavis.runners.morai_trainer import batch,autocast,objective,gradient_report
    item=batch(sample,device);model.train();model.zero_grad(set_to_none=True)
    with autocast(model):
        output=model(item['model_inputs'],retain_visual_grad=True)
        loss,_=objective(output,item['training_targets'],model.config)
    loss.backward();gradients=gradient_report(model)
    history_grad=output['visual_feature'].grad
    history_norm=None if history_grad is None else history_grad[:,:-1].float().norm().item()
    model.zero_grad(set_to_none=True);model.eval()
    original=item['model_inputs']
    def mutate_sensor(value,keys,frames=slice(None)):
        for key in keys:value['sensors'][key][:,frames].zero_()
    all_keys=('rgb_front','rgb_center','rgb_left','rgb_right','rgb_rear','lidar')
    with torch.no_grad(),autocast(model):
        baseline=model(original)
        changed=copy.deepcopy(original)
        mutate_sensor(changed,all_keys,-1);changed['route'][:,-1,:,1]+=5;changed['elapsed_s'][:,-1]+=.1
        future=model(changed)
        future_delta=max((baseline[k][:,:-1]-future[k][:,:-1]).abs().max().item() for k in ('trajectory','speed'))
        dependencies={}
        for name,keys in [('front',('rgb_front','rgb_center')),('left',('rgb_left',)),('right',('rgb_right',)),
                          ('rear',('rgb_rear',)),('lidar',('lidar',)),('history',all_keys)]:
            changed=copy.deepcopy(original)
            mutate_sensor(changed,keys,slice(None,-1) if name=='history' else slice(None))
            value=model(changed)
            dependencies[name]=max((value[k][:,-1]-baseline[k][:,-1]).abs().max().item() for k in ('trajectory','speed'))
        changed=copy.deepcopy(original);changed['route'][:,:,:,1]+=3
        value=model(changed)
        dependencies['route']=(value['trajectory'][:,-1]-baseline['trajectory'][:,-1]).abs().max().item()
    rejection={}
    for key in ('ego_teacher','future_trajectory','gps','source_stamp_ns'):
        try:model(dict(original,**{key:torch.zeros(1,device=device)}))
        except ValueError:rejection[key]=True
        else:rejection[key]=False
    passed=gradients['passed'] and future_delta<=1e-5 and all(v>0 for v in dependencies.values()) and all(rejection.values())
    return dict(passed=passed,gradient=gradients,future_to_past_max_delta=future_delta,
        dependencies=dependencies,rejected_inputs=rejection,past_visual_gradient_norm=history_norm,
        semantic_training=False,optimizer_updates=0)
