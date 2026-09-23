import unittest
import numpy as np
import morai_geometry as g

def camera(yaw=0):
    return dict(width=640,height=480,camera_model="pinhole",fov_axis="horizontal",
                fov_deg=90,position_m=[0,0,1.2],rotation_deg=[0,0,yaw])

class GeometryTests(unittest.TestCase):
    def test_front_ground_projection(self):
        q=g.camera_projection(camera())@np.array([10,0,0,1.])
        np.testing.assert_allclose(q[:2]/q[2],[320,278.4])
        self.assertGreater(q[2],0)
    def test_rear_depth(self):
        P=g.camera_projection(camera(180))
        self.assertGreater((P@[-10,0,0,1])[2],0)
        self.assertLess((P@[10,0,0,1])[2],0)
    def test_horizontal_fov(self):
        P=g.camera_projection(camera())
        q=P@np.array([10,10,1.2,1])
        np.testing.assert_allclose(q[:2]/q[2],[0,240],atol=1e-9)
    def test_ground_height(self):
        self.assertEqual((g.bev_world_matrix(-.4)@[256,128,1])[2],-.4)
    def test_visibility_rejects_behind_camera(self):
        cams={name:camera() for name in g.ORDER}
        H,uv,m=g.build_geometry(cams)
        self.assertFalse(m[0][:,:256].any())
        self.assertTrue(m[0][128,330])
        self.assertTrue(np.isfinite(H).all())

if __name__=="__main__":unittest.main()
