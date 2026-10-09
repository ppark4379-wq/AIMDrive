"""Clearly synthetic data for UI inspection only. No files or training writes."""
from datetime import datetime, timedelta, timezone
import math
import random


def demo_data():
    rng = random.Random(42)
    rows=[]; elapsed=0.; now=datetime.now(timezone.utc)
    for i in range(1,1201):
        elapsed += 1.82+rng.uniform(-.08,.1)
        keep = .30+.19*math.exp(-i/140)+rng.gauss(0,.009)
        rows.append(dict(update=i,total_updates=6000,total_epochs=15,epoch=(i-1)//400+1,
            timestamp=(now-timedelta(seconds=(1200-i)*1.83)).isoformat(),elapsed_s=elapsed,
            update_time_s=1.82+rng.uniform(-.08,.1),lr=1e-5*(1+math.cos(math.pi*i/6000))/2,
            loss=1.4+3*math.exp(-i/220)+rng.gauss(0,.08),waypoint_loss=.65+2.8*math.exp(-i/200)+rng.gauss(0,.09),
            pruning_loss=.003+.08*math.exp(-i/120)+abs(rng.gauss(0,.0005)),
            reconstruction_loss=.54+.8*math.exp(-i/160)+rng.gauss(0,.035),
            sampled_retention=keep,soft_retention=.31+.15*math.exp(-i/130)+rng.gauss(0,.002),
            p_keep_std=.07*(1-math.exp(-i/300))+.012+rng.gauss(0,.002),
            selected_tokens_mean=106*keep,selected_tokens_min=24,selected_tokens_max=43,
            selected_tokens_target=31.8,target_retention=.3,num_visual_tokens=106,
            grad_norm=3.2+rng.uniform(-.3,.3),gpu_memory_gb=6.38+rng.uniform(-.01,.01)))
        ade = .42 + 1.5*math.exp(-i/250)
        fde = ade * 1.65
        rows[-1].update(train_ade_m=ade, train_fde_m=fde, trajectory_samples=4,
            zone_metrics={str(i % 5): dict(samples=4, ade_sum_m=4*ade, fde_sum_m=4*fde)},
            navigation_tokens=220., situation_tokens=264., zone_tokens=11., visual_tokens=350.,
            llm_valid_tokens=845., llm_max_valid_tokens=910., llm_padded_tokens=1056,
            llm_padding_ratio=1-845/1056, llm_context_limit=2048)
        if i%100==0: rows[-1]['module_gradients']=dict(selector=1.42,mefa=.81,qformer=1.03,connector_projection=.67,tinyllama=.39,waypoint_head=1.21,navigation_mlp=.24,zone_embedding=.08)
    rows[0]['system']=dict(gpu='Example GPU (synthetic)',cuda='—',pytorch='—',python='—')
    return rows,dict(total_updates=6000,total_epochs=15,updates_per_epoch=400,target_retention=.3,
                     num_visual_tokens=106,validation_enabled=False,path_navigation=True,situation_instruction=True)


def demo_epochs():
    return [dict(epoch=e, updates=e*400, training=dict(ade_m=1.2/e, fde_m=2./e)) for e in (1, 2, 3)]
