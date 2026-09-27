"""ROS-free Open3D stage viewer in a separate spawned process."""
import colorsys
import multiprocessing as mp
import queue
import time
import numpy as np

STAGES = ("raw", "roi", "voxel", "ground", "nonground", "clusters")


def viewer_payload(snapshot):
    """Send point arrays once, not the full frame/cluster dataclass graph."""
    frame = snapshot["frame"]
    if snapshot.get("roi_only") or snapshot.get("voxel_only") or snapshot.get("ground_only"):
        stages = ("roi", "voxel") if snapshot.get("voxel_only") else ("roi",)
        if snapshot.get("ground_only"):
            stages = ("voxel", "ground", "nonground")
        return {"points": {key: np.ascontiguousarray(snapshot[key][:, :3], dtype=np.float32) for key in stages},
                "labels": np.empty(0, dtype=np.int32), "centroids": np.empty((0, 3), dtype=np.float32),
                "frame_id": frame.frame_id, "raw_timestamp": frame.raw_timestamp,
                "complete": frame.complete, "timestamp_status": frame.timestamp_status,
                "ground_status": frame.diagnostics.get("ground_status"),
                "ground_points": frame.diagnostics.get("ground_points", 0)}
    labels = np.zeros(len(snapshot["nonground"]), dtype=np.int32)
    for component, cluster in zip(snapshot["components"], snapshot["clusters"]):
        labels[component] = cluster.cluster_id + 1
    return {"points": {key: np.ascontiguousarray(snapshot[key][:, :3], dtype=np.float32)
                       for key in STAGES[:5]},
            "labels": labels,
            "centroids": np.asarray([c.centroid for c in snapshot["clusters"]],
                                    dtype=np.float32).reshape(-1, 3),
            "frame_id": frame.frame_id, "raw_timestamp": frame.raw_timestamp,
            "complete": frame.complete, "timestamp_status": frame.timestamp_status}


def stage_arrays(payload, stage, z_bounds):
    """Return full-resolution XYZ and RGB arrays; no display subsampling."""
    points = payload["points"]["nonground" if stage == "clusters" else stage]
    if stage == "clusters":
        labels = payload["labels"]
        palette = np.full((int(labels.max(initial=0)) + 1, 3), .25)
        for label in np.unique(labels):
            if label:
                palette[label] = colorsys.hsv_to_rgb(((int(label) - 1) * .61803398875) % 1, .8, 1)
        colors = palette[labels]
    elif stage == "ground":
        colors = np.tile([.7, .4, .16], (len(points), 1))
    else:
        t = np.clip((points[:, 2] - z_bounds[0]) / (z_bounds[1] - z_bounds[0]), 0, 1)
        colors = np.column_stack((t, 1 - np.abs(2 * t - 1), 1 - t))
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def centroid_crosses(centers, radius=.25):
    vertices, lines = [], []
    for center in centers:
        for axis in np.eye(3):
            offset = len(vertices)
            vertices.extend([center - radius * axis, center + radius * axis])
            lines.append([offset, offset + 1])
    return np.asarray(vertices).reshape(-1, 3), np.asarray(lines, dtype=np.int32).reshape(-1, 2)


def _viewer_loop(inbox, status, stop, config, initial_stage, visible, roi_only=False, voxel_only=False, ground_only=False):
    vis = None
    try:
        import open3d as o3d
        vis = o3d.visualization.VisualizerWithKeyCallback()
        title = ("LiDAR | ROI only | Space pause | N latest | F fit | Q quit" if roi_only else
                 "LiDAR | 1 raw 2 ROI 3 voxel 4 ground 5 nonground 6 clusters | Space pause | N latest | C centroids")
        if voxel_only:
            title = f"LiDAR | 1 ROI 2 voxel ({config['voxel_size']} m) | Space pause | N latest | F fit | Q quit"
        if ground_only:
            title = "LiDAR RANSAC | 1 before 2 removed ground 3 remaining | Space pause | N latest | F fit | Q quit"
        if not vis.create_window(title,
                                 width=1280, height=800, visible=visible):
            raise RuntimeError("Open3D could not create an OpenGL window")
        cloud, crosses = o3d.geometry.PointCloud(), o3d.geometry.LineSet()
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
        vis.add_geometry(axes)
        # Establish ROI-sized bounds before the first frame, avoiding per-frame zoom changes.
        bounds = o3d.geometry.AxisAlignedBoundingBox(config["roi_min"], config["roi_max"])
        box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bounds)
        box.paint_uniform_color([.3, .3, .3])
        vis.add_geometry(box)
        cloud.points = o3d.utility.Vector3dVector([[0, 0, 0]])
        crosses.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0]])
        crosses.lines = o3d.utility.Vector2iVector([[0, 1]])
        vis.add_geometry(cloud, reset_bounding_box=False)
        vis.add_geometry(crosses, reset_bounding_box=False)
        options = vis.get_render_option()
        options.background_color = np.array([.04, .04, .06])
        options.point_size = 2
        state = {"stage": initial_stage, "paused": False, "advance": False,
                 "centroids": True, "dirty": True}
        def choose(stage):
            def callback(_):
                state.update(stage=stage, dirty=True)
                return False
            return callback
        stages = ("voxel", "ground", "nonground") if ground_only else (("roi", "voxel") if voxel_only else STAGES)
        for index, name in enumerate(stages, 1):
            if not roi_only or name == "roi":
                vis.register_key_callback(ord(str(index)), choose(name))
        def pause(_):
            state["paused"] = not state["paused"]
            state["dirty"] = True
            return False
        def advance(_):
            state.update(paused=True, advance=True)
            return False
        def centers(_):
            state["centroids"] = not state["centroids"]
            state["dirty"] = True
            return False
        def quit_viewer(_):
            stop.set()
            return False
        def fit(_):
            vis.reset_view_point(reset_bounding_box=True)
            return False
        vis.register_key_callback(32, pause)
        vis.register_key_callback(ord("N"), advance)
        vis.register_key_callback(ord("C"), centers)
        vis.register_key_callback(ord("Q"), quit_viewer)
        vis.register_key_callback(ord("F"), fit)
        status.put(("ready", None))
        current = latest = None
        first_render = True
        last_report = 0
        while not stop.is_set() and vis.poll_events():
            try:
                latest = inbox.get_nowait()
            except queue.Empty:
                pass
            if latest is not None and (not state["paused"] or state["advance"] or current is None):
                current, latest = latest, None
                state.update(dirty=True, advance=False)
            if current is not None and state["dirty"]:
                points, colors = stage_arrays(current, state["stage"],
                    (config["roi_min"][2], config["roi_max"][2]))
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                centers_xyz = current["centroids"] if state["centroids"] and state["stage"] == "clusters" else np.empty((0, 3))
                vertices, lines = centroid_crosses(centers_xyz)
                crosses.points = o3d.utility.Vector3dVector(vertices)
                crosses.lines = o3d.utility.Vector2iVector(lines)
                crosses.paint_uniform_color([1, 1, 1])
                vis.update_geometry(cloud)
                vis.update_geometry(crosses)
                if first_render:
                    status.put(("rendered", current["frame_id"]))
                    first_render = False
                state["dirty"] = False
                if time.monotonic() - last_report >= 1 or state["paused"]:
                    print(f"[viewer] stage={state['stage']} frame={current['frame_id']} "
                          f"points={len(points)} clusters={len(current['centroids'])} "
                          f"complete={current['complete']} paused={state['paused']}"
                          + (f" ground_status={current['ground_status']} removed={current['ground_points']}" if ground_only else ""), flush=True)
                    last_report = time.monotonic()
            vis.update_renderer()
            time.sleep(.005)
    except Exception as exc:
        status.put(("error", str(exc)))
    finally:
        if vis is not None:
            vis.destroy_window()
        stop.set()


class DirectStageViewer:
    def __init__(self, config, stage="clusters", visible=True, roi_only=False, voxel_only=False, ground_only=False):
        if sum((roi_only, voxel_only, ground_only)) > 1:
            raise ValueError("Choose one preview stage")
        if ground_only:
            stage = "nonground"
        if voxel_only:
            stage = "voxel"
        if roi_only:
            stage = "roi"
        if stage not in STAGES:
            raise ValueError("Unknown point-cloud stage")
        context = mp.get_context("spawn")
        self.inbox = context.Queue(maxsize=1)
        self.status = context.Queue()
        self.stop = context.Event()
        self.dropped_frames = 0
        self.process = context.Process(target=_viewer_loop,
            args=(self.inbox, self.status, self.stop, config, stage, visible, roi_only, voxel_only, ground_only), daemon=True)
        self.process.start()
        try:
            kind, message = self.status.get(timeout=45)
            if kind != "ready":
                raise RuntimeError(message)
        except BaseException:
            self.close()
            raise

    def update(self, snapshot):
        try:
            kind, message = self.status.get_nowait()
            if kind == "error":
                raise RuntimeError(f"Open3D viewer: {message}")
        except queue.Empty:
            pass
        if self.stop.is_set():
            return
        if not self.process.is_alive():
            raise RuntimeError(f"Open3D viewer exited unexpectedly ({self.process.exitcode})")
        try:
            self.inbox.put_nowait(viewer_payload(snapshot))
        except queue.Full:
            # Rendering can skip frames; UDP/processing must not wait for the GUI.
            self.dropped_frames += 1

    def close(self):
        self.stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.inbox.cancel_join_thread()
        self.inbox.close()
        self.status.close()
