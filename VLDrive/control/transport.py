"""Session-scoped UDP client and owned local Windows bridge launcher."""
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import uuid

from control.protocol import encode_message

ROOT=Path(__file__).resolve().parents[1]


def windows_host():
    if sys.platform=='win32': return '127.0.0.1'
    for line in Path('/proc/net/route').read_text().splitlines()[1:]:
        fields=line.split()
        if fields[1]=='00000000' and int(fields[3],16)&2:
            return socket.inet_ntoa(struct.pack('<I',int(fields[2],16)))
    raise RuntimeError('Cannot determine Windows host; set bridge.host explicitly')


class BridgeClient:
    def __init__(self,cfg,config_path,output):
        self.options=cfg['bridge'];self.session=uuid.uuid4().hex;self.sequence=0
        host=windows_host() if self.options['host']=='windows_host' else self.options['host']
        self.target=(socket.gethostbyname(host),self.options['listen_port'])
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0',0));self.socket.setblocking(False)
        self.last_ack=None;self.last_ack_sequence=0;self.sent=0;self.process=None;self.connected=False
        self.output=Path(output)
        try:
            if self.options['auto_start']:
                self._launch(Path(config_path))
            deadline=time.monotonic()+5.
            while time.monotonic()<deadline:
                self._message('hello');time.sleep(.1);self.poll()
                if self.last_ack is not None:
                    self.connected=True
                    return
            raise RuntimeError('Windows control bridge did not acknowledge; check bridge host/port/firewall and bridge.stderr.log')
        except BaseException:
            self.socket.close()
            raise

    def _launch(self,config_path):
        script=ROOT/'control/udp_bridge.py'
        report=self.output/'bridge_report.json'
        if sys.platform=='win32':
            args=[sys.executable,str(script),'--config',str(config_path),'--session',self.session,'--report',str(report)]
            extra=dict(creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            python=ROOT/self.options['windows_python']
            if not python.is_file() or not Path('/init').is_file():
                raise RuntimeError('Configured Windows pythonw/WSL interop unavailable')
            def win(path):
                return subprocess.check_output(['wslpath','-w',str(path.resolve())],text=True,timeout=5).strip()
            args=['/init',str(python),win(python),win(script),'--config',win(config_path),
                  '--session',self.session,'--report',win(report)]
            extra={}
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as route:
            route.connect(self.target)  # UDP routing lookup; sends no datagram.
            client_host=route.getsockname()[0]
        args+=['--client-host',client_host,'--client-port',str(self.socket.getsockname()[1])]
        with (self.output/'bridge.stdout.log').open('xb') as out,(self.output/'bridge.stderr.log').open('xb') as err:
            self.process=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=out,stderr=err,**extra)

    def _message(self,kind,command=None):
        self.sequence+=1
        self.socket.sendto(encode_message(self.session,self.sequence,kind,command),self.target)
        self.sent+=1

    def poll(self):
        for _ in range(64):
            try:
                payload,peer=self.socket.recvfrom(2048)
            except BlockingIOError:
                break
            try:
                ack=json.loads(payload)
                if (peer==self.target and ack['session']==self.session and ack['accepted'] is True
                        and type(ack['sequence']) is int and self.last_ack_sequence<ack['sequence']<=self.sequence):
                    self.last_ack=time.monotonic()
                    self.last_ack_sequence=ack['sequence']
            except (ValueError,KeyError,TypeError):
                pass

    def send_command(self,command):
        self._message('command',command);self.poll()

    def close(self):
        try:
            if self.connected:
                for _ in range(3):
                    self._message('stop');time.sleep(.02)
        except OSError:
            pass  # Independent Windows watchdog still brakes on loss of WSL.
        finally:
            self.connected=False
            self.socket.close()
