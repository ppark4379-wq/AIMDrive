"""MORAI pinhole projection to the existing 1_FRLR BEV coordinate system."""
import numpy as np

ORDER = ("front", "rear", "left", "right")
SHAPE = (256, 512)
# Preserve the trained drone projection and its central crop (604x964 -> 256x512).
METERS_PER_PIXEL = (50.0 / 682.578) * (964.0 / 512.0)


def camera_projection(camera):
    if camera["camera_model"] != "pinhole" or camera["fov_axis"] != "horizontal":
        raise ValueError("Expected confirmed pinhole camera and horizontal FOV")
    w, h = camera["width"], camera["height"]
    f = w / (2 * np.tan(np.deg2rad(camera["fov_deg"]) / 2))
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]])
    roll, pitch, yaw = np.deg2rad(camera["rotation_deg"])
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    # Assumption: vehicle X forward, Y left, Z up; positive pitch points down.
    # Camera local FLU axes rotated by Rz(yaw) Ry(pitch) Rx(roll).
    optical = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    R = optical @ (Rz @ Ry @ Rx).T
    center = np.array(camera["position_m"])
    return K @ np.column_stack((R, -R @ center))


def crop_matrix(height, width):
    # Match utils.resize_image_op exactly, including integer crop coordinates.
    target_h, target_w = SHAPE
    if target_w / width < target_h / height:
        crop_w = int(height * target_w / target_h)
        offset_x, offset_y = int((width - height * target_w / target_h)/2), 0
        crop_h = height
    else:
        crop_h = int(width * target_h / target_w)
        offset_y, offset_x = int((height - width * target_h / target_w)/2), 0
        crop_w = width
    return np.array([[target_w/crop_w,0,-offset_x*target_w/crop_w],
                     [0,target_h/crop_h,-offset_y*target_h/crop_h],[0,0,1.]])


def bev_world_matrix(ground_z=0.0):
    h,w = SHAPE
    s = METERS_PER_PIXEL
    return np.array([[s,0,-w*s/2],[0,-s,h*s/2],[0,0,ground_z],[0,0,1.]])


def build_geometry(cameras, ground_z=0.0):
    h,w = SHAPE
    pixels_from_unit = np.array([[w/2,0,w/2],[0,h/2,h/2],[0,0,1.]])
    unit_from_pixels = np.linalg.inv(pixels_from_unit)
    world = bev_world_matrix(ground_z)
    v,u = np.mgrid[:h,:w]
    grid = np.stack((u,v,np.ones_like(u))).reshape(3,-1)
    matrices, coordinates, valid = [], [], []
    for name in ORDER:
        camera = cameras[name]
        projection = camera_projection(camera) @ world
        projected = projection @ grid
        depth = projected[2]
        uv = projected[:2] / np.where(abs(depth)>1e-8, depth, 1.0)
        uv[:,abs(depth)<=1e-8] = -1e6
        network_uv = crop_matrix(camera["height"],camera["width"]) @ np.vstack((uv,np.ones(h*w)))
        visible = ((depth>1e-6) & (uv[0]>=0) & (uv[0]<camera["width"]-1)
                   & (uv[1]>=0) & (uv[1]<camera["height"]-1)
                   & (network_uv[0]>=0) & (network_uv[0]<w-1)
                   & (network_uv[1]>=0) & (network_uv[1]<h-1))
        matrices.append(unit_from_pixels @ crop_matrix(camera["height"],camera["width"])
                        @ projection @ pixels_from_unit)
        coordinates.append(uv.reshape(2,h,w).astype(np.float32))
        valid.append(visible.reshape(h,w))
    return matrices, coordinates, valid
