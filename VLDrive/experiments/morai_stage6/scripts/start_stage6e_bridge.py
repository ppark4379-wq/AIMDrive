#!/usr/bin/env python3
"""Start one Stage6E binary-JSON-compatible bridge and, if needed, a local master.

Sources the configured ROS/catkin setup with system Python. Reuses an existing master, refuses an
occupied bridge port, and only stops children it started. Logs stay in this repo.
"""
import argparse
import datetime
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse
import xmlrpc.client

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master',default=os.environ.get('ROS_MASTER_URI','http://localhost:11311'))
    parser.add_argument('--port',type=int,default=9090)
    parser.add_argument('--ros-setup',default=str(Path.home()/'catkin_ws/devel/setup.bash'))
    args=parser.parse_args()
    if os.environ.get('STAGE6E_ROS_BOOTSTRAPPED')!='1':
        setup=Path(args.ros_setup).expanduser().resolve()
        if not setup.is_file():
            parser.error('Missing ROS/catkin setup: '+str(setup))
        env=dict(os.environ,STAGE6E_ROS_BOOTSTRAPPED='1',PYTHONDONTWRITEBYTECODE='1')
        # Always use Noetic's system Python, including when invoked from conda.
        os.execvpe('bash',['bash','-c','source "$1"; shift; exec /usr/bin/python3 "$@"',
                           'stage6e',str(setup),str(Path(__file__).resolve())]+sys.argv[1:],env)
    with socket.socket() as check:
        # A closed bridge may leave TIME_WAIT sockets. Match the server's bind
        # semantics; an active listener still conflicts (no SO_REUSEPORT).
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            check.bind(('0.0.0.0',args.port))
        except OSError:
            parser.error('Bridge port %d is already in use. Do not launch a second bridge.'%args.port)
    output=ROOT/'diagnostics'/('bridge_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(ROS_MASTER_URI=args.master,ROS_HOME=str(output/'ros_home'),
               ROS_LOG_DIR=str(output/'ros_logs'),PYTHONDONTWRITEBYTECODE='1')
    children=[]
    log=None
    socket.setdefaulttimeout(3)
    def master_ready():
        try:
            return xmlrpc.client.ServerProxy(args.master).getPid('/stage6e_launcher')[0]==1
        except (OSError,xmlrpc.client.Error):
            return False
    try:
        if not master_ready():
            url=urlparse(args.master)
            # A bind proves this is a local interface. Never start a different
            # master silently when the configured remote master is unavailable.
            try:
                with socket.socket() as check:
                    check.bind((url.hostname,0))
            except OSError:
                raise RuntimeError('Configured remote master unavailable: '+args.master)
            log=(output/'roscore.log').open('x')
            core=subprocess.Popen([sys.executable,'/opt/ros/noetic/bin/roscore','-p',str(url.port or 11311)],
                                  cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(core)
            deadline=time.monotonic()+20
            while not master_ready():
                if core.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Master did not start; see '+str(output/'roscore.log'))
                time.sleep(.2)
        bridge=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_stage6e_bridge.py'),
                                 '--port',str(args.port),'__name:=stage6e_morai_bridge'],cwd=str(ROOT),env=env)
        children.append(bridge)
        print('Evidence:',output,flush=True)
        print('Connect MORAI sensors and vehicle information to this host, port',args.port,flush=True)
        print('Recorder verifies /stage6e/bridge_health automatically. Ctrl-C stops only this launcher\'s children.',flush=True)
        return bridge.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
        if log:
            log.close()


if __name__=='__main__':
    raise SystemExit(main())
