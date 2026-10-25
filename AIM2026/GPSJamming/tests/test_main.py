"""Exercise the new entrypoint through real loopback sensor and boundary sockets."""
from pathlib import Path
import select
import socket
import subprocess
import sys
import time
import unittest

from test_lidar import packet
from src.UDP import WallUDPReceiver

ROOT = Path(__file__).resolve().parents[1]


class MainTests(unittest.TestCase):
    def test_no_ros_dependency_and_cli_help(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import main,sys; assert 'rospy' not in sys.modules; main.main(['--help'])"],
            cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--boundary-port', result.stdout)

    def test_sensor_udp_to_planning_udp_without_ros(self):
        # WSL mirrored networking can intercept/fragment large 127.0.0.1 datagrams.
        # Use the Linux loopback alias for this process-to-process integration test.
        sink = WallUDPReceiver(bind='127.0.0.2', port=0)
        sensor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sensor.bind(('127.0.0.1', 0))
        input_port = sensor.getsockname()[1]
        sensor.close()
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        child = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT/'main.py'),
            '--bind', '127.0.0.1', '--port', str(input_port),
            '--boundary-host', '127.0.0.2',
            '--boundary-port', str(sink.socket.getsockname()[1]), '--duration', '1.5'],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([child.stdout], [], [], 5)[0], 'No startup message')
            self.assertIn('LiDAR UDP', child.stdout.readline())
            for start in range(75):
                sender.sendto(packet(123456, start*4.8, step=.4), ('127.0.0.1', input_port))
            deadline = time.monotonic() + 3
            result = None
            while result is None and time.monotonic() < deadline:
                result = sink.poll()
                time.sleep(.01)
            if result is None:
                output, errors = child.communicate(timeout=5)
                self.fail('No planning UDP packet; stdout=%s stderr=%s' % (output, errors))
            boundary = result['boundaries']
            self.assertEqual(boundary['raw_timestamp'], 123456)
            self.assertEqual(boundary['packet_count'], 75)
            self.assertEqual(boundary['frame_id'], 'base_link')
            self.assertTrue(boundary['complete'])
            self.assertEqual(boundary['stamp_source'], 'system_publish_time')
            output, errors = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, errors)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()
            sender.close()
            sink.close()
