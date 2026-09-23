"""VLDrive labels and camera layout around the existing paper waypoint plot."""
import textwrap

import numpy as np

from utils import vision_encoder_visualization as viz


def render_vldrive_waypoints(waypoints, *, instruction, frame_id, route, selected_tokens,
                            camera_image=None, actual_trajectory=None, example=False):
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.shape != (5,2) or not np.isfinite(waypoints).all():
        raise ValueError('VLDrive waypoints must be finite [5,2]')
    width = 92 if camera_image is not None else 58
    caption = '\n'.join(textwrap.fill(line,width=width) for line in (
        f'{route} | frame {frame_id}', 'Instruction: ' + instruction,
        f'CCDP retained: {selected_tokens} / 106' if selected_tokens is not None else 'No model was loaded.'))
    title = 'VLDrive predicted waypoints'
    if example:
        title = 'Renderer example (synthetic waypoints)'
        caption = 'LAYOUT TEST ONLY - NOT VLDRIVE PREDICTIONS\n' + caption
    # Reuse all axes, ego footprint, horizon labels, metric scale and styling.
    figure = viz.render_waypoints(waypoints, title=title, subtitle=caption, figsize=(5.,6.4))
    axes, colorbar_axes = figure.axes
    if actual_trajectory is not None:
        actual = np.asarray(actual_trajectory,dtype=np.float64)
        if actual.shape != (5,2) or not np.isfinite(actual).all():
            raise ValueError('Actual trajectory must be finite [5,2]')
        forward,right = viz.forward_right(actual)
        axes.plot(np.r_[0.,right],np.r_[0.,forward],color=viz.MUTED,linestyle='--',
                  linewidth=1.2,marker='x',markersize=4,label='Actual future ego trajectory',zorder=2)
        axes.lines[0].set_label('VLDrive predicted waypoints')
        xmin,xmax = axes.get_xlim()
        ymin,ymax = axes.get_ylim()
        axes.set_xlim(min(xmin,float(right.min())-2),max(xmax,float(right.max())+2))
        axes.set_ylim(min(ymin,float(forward.min())-2),max(ymax,float(forward.max())+5))
        axes.legend(loc='upper left',fontsize=7)
    if camera_image is not None:
        # Keep the BEV and colorbar as vector artists, not a screenshot of a plot.
        figure.set_layout_engine('none')
        figure.set_size_inches(10.,6.2)
        axes.set_position([.63,.27,.29,.62])
        colorbar_axes.set_position([.94,.27,.015,.62])
        camera_axes = figure.add_axes([.025,.34,.53,.53])
        camera_axes.imshow(camera_image)
        camera_axes.set_xticks([])
        camera_axes.set_yticks([])
        camera_axes.set_title('Front camera (anchor frame)',loc='left',pad=6)
        for spine in camera_axes.spines.values():
            spine.set_linewidth(.8)
        axes.set_title('Future waypoints (ego frame)',loc='left')
        figure.suptitle(title,fontsize=11,y=.97)
        figure.supxlabel('path length {:.1f} m\n{}'.format(
            np.linalg.norm(np.diff(np.vstack([np.zeros((1,2)),waypoints]),axis=0),axis=1).sum(),caption),
            fontsize=7.5,color=viz.MUTED,ha='left',x=.025,y=.02)
    return figure
