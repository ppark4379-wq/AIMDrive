"""Independent axes and restrained, export-friendly scientific plots."""
import math
import plotly.graph_objects as go
from utils.visualization.training_dashboard.metrics import ema

BLUE = '#285f91'


def chart(rows, key, title, *, percent=False, target=None, span=100, height=218):
    x = [r['update'] for r in rows]
    y = [r.get(key) for r in rows]
    smooth = ema(y, span)
    # Retain raw extrema in each bucket when displaying long runs. CSV is full resolution.
    stride = max(1, math.ceil(len(rows)/900))
    keep = set([0, len(rows)-1]) if rows else set()
    for start in range(0,len(rows),stride):
        group = [i for i in range(start,min(start+stride,len(rows))) if y[i] is not None]
        if group: keep.update((group[0],group[-1],min(group,key=lambda i:y[i]),max(group,key=lambda i:y[i])))
    idx = sorted(keep)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x[i] for i in idx],y=[y[i] for i in idx],mode='lines',name='Raw',
                            line=dict(color='rgba(40,95,145,0.23)',width=1),connectgaps=False))
    fig.add_trace(go.Scatter(x=[x[i] for i in idx],y=[smooth[i] for i in idx],mode='lines',name=f'EMA {span}',
                            line=dict(color=BLUE,width=2),connectgaps=False))
    if target is not None:
        fig.add_hline(y=target,line_dash='dash',line_color='#737d87',line_width=1)
    if not any(v is not None for v in y):
        fig.add_annotation(text='Not recorded in this log',x=.5,y=.5,xref='paper',yref='paper',showarrow=False,
                           font=dict(color='#7d858e',size=12))
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=50,r=18,t=30,b=42),
        paper_bgcolor='white',plot_bgcolor='white',font=dict(family='Arial, sans-serif',size=11,color='#343b43'),
        title=dict(text=title,font=dict(size=13),x=.02),showlegend=False,
        hovermode='x unified',uirevision=key)
    fig.update_xaxes(title_text='Training update',gridcolor='#edf0f2',zeroline=False,
                     linecolor='#d5dbe0',showline=True,ticks='outside',tickcolor='#cbd2d9')
    fig.update_yaxes(gridcolor='#edf0f2',zeroline=False,tickformat='.0%' if percent else '.3~g',
                     linecolor='#d5dbe0',showline=True,ticks='outside',tickcolor='#cbd2d9')
    return fig
