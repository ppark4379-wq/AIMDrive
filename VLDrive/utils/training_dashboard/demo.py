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
        if i%100==0: rows[-1]['module_gradients']=dict(selector=1.42,mefa=.81,qformer=1.03,connector_projection=.67,tinyllama=.39,waypoint_head=1.21)
    rows[0]['system']=dict(gpu='Example GPU (synthetic)',cuda='—',pytorch='—',python='—')
    return rows,dict(total_updates=6000,total_epochs=15,updates_per_epoch=400,target_retention=.3,num_visual_tokens=106)
