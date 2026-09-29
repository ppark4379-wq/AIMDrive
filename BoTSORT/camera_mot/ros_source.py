"""Read CompressedImage messages; split a shared topic by header.frame_id."""
import time
from camera_mot.streams import CameraBuffers


class RosCameraSource(CameraBuffers):
    def __init__(self, cameras, topic):
        super().__init__(cameras, 'ros', 'morai_ros_header_raw')
        try:
            import rospy
            from sensor_msgs.msg import CompressedImage
        except ImportError as exc:
            raise RuntimeError('Source /opt/ros/noetic/setup.bash and install the WSL requirements') from exc
        self.rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node('aimdrive_camera_mot', anonymous=True, disable_signals=True)
        self.subscriber = rospy.Subscriber(topic, CompressedImage, self._receive,
            queue_size=30, buff_size=16 * 1024 * 1024, tcp_nodelay=True)

    def _receive(self, msg):
        self.put(msg.header.frame_id, int(msg.header.stamp.secs),
            int(msg.header.stamp.nsecs), bytes(msg.data), time.monotonic_ns())

    def close(self):
        self.subscriber.unregister()
        super().close()
