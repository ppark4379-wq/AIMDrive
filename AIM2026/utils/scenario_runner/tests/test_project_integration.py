# Moved runtime/data paths must never require a neighbouring project copy.
from pathlib import Path
import unittest

from scenario_runner.project_paths import (AIM_ROOT, RUNNER_ROOT, SDK_SRC, GEOMETRY_ROOT,
    project_path, morai_path, logical_source_path)


class NativeProjectPathsTest(unittest.TestCase):
    def test_original_sdk_and_measurement_identities_resolve_to_existing_bytes(self):
        cases = (
            ('data_collection/third_party/grpc_inha_univ/src/api/morai_sim_client.py',
             SDK_SRC / 'api/morai_sim_client.py'),
            ('data_collection/internal_sensor_record/evidence/object_geometry_registry_default3d_v2/registry.json',
             GEOMETRY_ROOT / 'object_geometry_registry_default3d_v2/registry.json'),
        )
        for logical, actual in cases:
            with self.subTest(logical=logical):
                self.assertTrue(actual.is_file())
                self.assertEqual(morai_path(logical), actual)
                self.assertEqual(logical_source_path(actual), logical)
                recorded = '/home/jun/AIMDrive/AIM2026/_support/MoraiDrive/' + logical
                self.assertEqual(project_path(recorded), actual)

    def test_parent_relative_legacy_path_uses_the_same_native_mapping(self):
        self.assertEqual(project_path('../MoraiDrive/scenario_runner/config/global.yaml'),
                         RUNNER_ROOT / 'config/global.yaml')
        self.assertEqual(project_path('../MoraiDrive/expert/src/morai_msgs/package.xml'),
                         AIM_ROOT / 'Expert/ros/src/morai_msgs/package.xml')
        self.assertEqual(project_path('../VLDrive/inference/ve_objects_udp.py'),
                         AIM_ROOT.parent / 'VLDrive/inference/ve_objects_udp.py')
        with self.assertRaises(ValueError):
            project_path('../MoraiDrive/data_collection/internal_sensor_record/runtime_paths.py')

    def test_explicit_external_input_is_preserved(self):
        self.assertEqual(project_path('/tmp/operator-provided-route.csv'),
                         Path('/tmp/operator-provided-route.csv'))


# Offline checks for native receiver, relay and process environment boundaries.
from collections import deque, Counter
import json
import math
import signal
import socket
import struct
import threading
from types import SimpleNamespace

import pytest

from scenario_runner.adapters.ros import binary_transport
from scenario_runner.utils import environment


















def test_analysis_python_uses_explicit_then_local_then_linux_candidates(monkeypatch,tmp_path):
    monkeypatch.setenv('MORAI_LOCAL_CONFIG',str(tmp_path/'machine.json'))
    monkeypatch.delenv('MORAI_ANALYSIS_PYTHON',raising=False)
    monkeypatch.delenv('AIM_EXPERT_PYTHON',raising=False)
    monkeypatch.setattr(environment.Path,'home',classmethod(lambda cls: tmp_path))
    path = tmp_path/'miniconda3/envs/lmdrive/bin/python3.8'
    path.parent.mkdir(parents=True)
    path.touch()
    assert environment.analysis_python() == path
    first = path.with_name('python')
    first.touch()
    assert environment.analysis_python() == first
    (tmp_path/'machine.json').write_text(json.dumps(dict(analysis_python='/tmp/explicit_local_python')))
    assert environment.analysis_python() == Path('/tmp/explicit_local_python')
    monkeypatch.setenv('MORAI_ANALYSIS_PYTHON','/tmp/explicit_env_python')
    assert environment.analysis_python() == Path('/tmp/explicit_env_python')


def test_analysis_python_does_not_spawn_to_select_interpreter(monkeypatch,tmp_path):
    monkeypatch.setenv('MORAI_LOCAL_CONFIG',str(tmp_path/'missing.json'))
    monkeypatch.delenv('MORAI_ANALYSIS_PYTHON',raising=False)
    monkeypatch.delenv('AIM_EXPERT_PYTHON',raising=False)
    monkeypatch.setattr(environment.subprocess,'run',lambda *_a,**_kw: pytest.fail('selection spawned a process'))
    assert isinstance(environment.analysis_python(),Path)


def test_ros_node_ip_uses_current_route_instead_of_inherited_ros_ip(monkeypatch):
    monkeypatch.setenv('ROS_IP','192.0.2.90')
    monkeypatch.delenv('MORAI_ROS_IP',raising=False)
    monkeypatch.setattr(environment.subprocess,'check_output',lambda *_a,**_kw: '1.1.1.1 via 172.20.0.1 dev eth0 src 172.20.2.3 uid 1000')
    assert environment.ros_ip() == '172.20.2.3'
    monkeypatch.setenv('MORAI_ROS_IP','192.0.2.45')
    assert environment.ros_ip() == '192.0.2.45'
    monkeypatch.setenv('MORAI_ROS_IP','127.0.0.1')
    with pytest.raises(ValueError):
        environment.ros_ip()


class FakeSocket:
    def __init__(self, data):
        self.chunks = iter(data)
        self.written = []
    def sendall(self,data):
        self.written.append(data)
    def recv(self,size):
        item = next(self.chunks,b'')
        if isinstance(item,Exception):
            raise item
        return item


def test_relay_frame_partial_timeout_and_multiple_messages():
    connection = FakeSocket([])
    binary_transport.send_frame(connection,{'topic':'camera'},b'jpeg')
    binary_transport.send_frame(connection,{'topic':'lidar'},b'cloud')
    data = b''.join(connection.written)
    connection.chunks = iter([data[:3],socket.timeout(),data[3:11],data[11:]])
    decoded = binary_transport.frames(connection,threading.Event())
    assert next(decoded) == ({'topic':'camera'},b'jpeg')
    assert next(decoded) == ({'topic':'lidar'},b'cloud')
    with pytest.raises(EOFError):
        next(decoded)


def test_relay_size_bound_checked_before_payload_read():
    connection = FakeSocket([struct.pack('!II',binary_transport.MAX_JSON+1,0)])
    with pytest.raises(ValueError,match='Oversize'):
        next(binary_transport.frames(connection,threading.Event()))






def test_child_environment_replaces_old_ros_prefixes_and_filters_only_project_libraries(monkeypatch):
    from scenario_runner.project_paths import AIM_ROOT
    base = dict(PYTHONHOME='/old/python',PYTHONPATH='/old/pythonpath',
        ROS_PACKAGE_PATH='/home/jun/AIMDrive/MoraiDrive/expert/src',
        CMAKE_PREFIX_PATH='/home/jun/AIMDrive/MoraiDrive/expert/devel',
        LD_LIBRARY_PATH='/home/jun/AIMDrive/MoraiDrive/expert/devel/lib:/opt/cuda/lib64:/home/jun/AIMDrive/AIM2026/_support/native/lib')
    env = environment.build_capture_environment(base)
    assert 'PYTHONHOME' not in env
    assert env['ROS_PACKAGE_PATH'] == str(environment.CATKIN_ROOT/'src')+':'+str(environment.ROS_ROOT/'share')
    assert env['CMAKE_PREFIX_PATH'] == str(environment.CATKIN_ROOT/'devel')+':'+str(environment.ROS_ROOT)
    assert '/opt/cuda/lib64' in env['LD_LIBRARY_PATH']
    assert not any(name in env['LD_LIBRARY_PATH'] for name in ('MoraiDrive','VLDrive','_support'))
    assert '/old/pythonpath' not in env['PYTHONPATH']










def test_shell_ros_setup_cannot_traverse_an_inherited_project_prefix(tmp_path):
    import os
    import subprocess
    from scenario_runner.project_paths import AIM_ROOT,RUNNER_ROOT
    setup=tmp_path/'setup.bash'
    setup.write_text('[[ -z "${CMAKE_PREFIX_PATH:-}" ]] || return 91\n[[ -z "${ROS_PACKAGE_PATH:-}" ]] || return 92\n')
    env=dict(os.environ,MORAI_ROS_ROOT=str(tmp_path),MORAI_CATKIN_WS=str(AIM_ROOT/'Expert/ros'),
        MORAI_ROS_IP='192.0.2.45',ROS_IP='192.0.2.90',
        CMAKE_PREFIX_PATH='/home/jun/AIMDrive/MoraiDrive/expert/devel',
        ROS_PACKAGE_PATH='/home/jun/AIMDrive/MoraiDrive/expert/src',
        PYTHONPATH='/home/jun/AIMDrive/VLDrive',
        LD_LIBRARY_PATH='/home/jun/AIMDrive/MoraiDrive/expert/devel/lib:/opt/cuda/lib64')
    script='source "$1"\nmorai_load_internal_sensor_record_environment\nprintf "%s\\n" "$ROS_PACKAGE_PATH" "$CMAKE_PREFIX_PATH" "$LD_LIBRARY_PATH" "$PYTHONPATH" "$ROS_IP"\n'
    completed=subprocess.run(['bash','-eu','-c',script,'test',str(RUNNER_ROOT/'tools/project_env.sh')],
        env=env,capture_output=True,text=True,timeout=10)
    assert completed.returncode == 0,completed.stderr
    lines=completed.stdout.splitlines()
    assert lines[0] == str(AIM_ROOT/'Expert/ros/src')+':'+str(tmp_path/'share')
    assert lines[1] == str(AIM_ROOT/'Expert/ros/devel')+':'+str(tmp_path)
    assert '/opt/cuda/lib64' in lines[2]
    assert not any(old in line for line in lines for old in ('MoraiDrive','VLDrive','_support'))
    assert lines[-1] == '192.0.2.45'










if __name__ == "__main__":
    unittest.main()
