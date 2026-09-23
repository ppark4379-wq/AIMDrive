#!/usr/bin/env python3
"""Windows bridge to MORAI localhost; command watchdog survives WSL failure."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from control.config import load_config
from control.protocol import decode_message,pack_command

STOP=dict(accel=0.,brake=1.,steer=0.)


class ControlBridge:
    def __init__(self,cfg,session,bootstrap=None):
        self.cfg=cfg;self.options=cfg['bridge'];self.session=session
        self.bootstrap=bootstrap
        self.receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.receiver.bind((self.options['listen_host'],self.options['listen_port']))
        self.receiver.settimeout(1/cfg['runtime']['send_hz'])
        self.sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.target=(self.options['target_host'],self.options['target_port'])
        self.report=dict(received=0,rejected=0,forwarded=0,watchdog_brakes=0,target=list(self.target))
        self.active=False

    def transmit(self,command):
        self.sender.sendto(pack_command(command),self.target)
        self.report['forwarded']+=1

    def run(self):
        started=time.monotonic();last_command=None;last_seq=-1;peer=None;last_brake=0.;last_probe=0.
        try:
            while True:
                # Initiate from Windows so WSL replies use the existing UDP flow.
                # This avoids opening a Windows inbound firewall rule.
                if peer is None and self.bootstrap is not None and time.monotonic()-last_probe>=.1:
                    self.receiver.sendto(json.dumps(dict(session=self.session,kind='ready')).encode(),self.bootstrap)
                    last_probe=time.monotonic()
                message=None
                try:
                    data,address=self.receiver.recvfrom(4096)
                    message=decode_message(data,self.session)
                    if self.bootstrap is not None and address!=self.bootstrap:
                        raise ValueError('Different bootstrap peer')
                    if message['sequence']<=last_seq or (peer is not None and address!=peer):
                        raise ValueError('Old sequence or different peer')
                    if peer is None and message['kind']!='hello':
                        raise ValueError('Handshake required')
                    peer=address;last_seq=message['sequence']
                    self.report['received']+=1
                    if message['kind']=='command':
                        self.transmit(message['command'])
                        self.active=True;last_command=time.monotonic()
                    elif message['kind']=='stop' and self.active:
                        self.transmit(STOP)
                    reply=dict(session=self.session,sequence=last_seq,accepted=True)
                    self.receiver.sendto(json.dumps(reply).encode(),address)
                    if message['kind']=='stop': break
                except socket.timeout:
                    pass
                except (ValueError,KeyError,TypeError,UnicodeError):
                    self.report['rejected']+=1
                now=time.monotonic()
                if last_command is None:
                    if now-started>self.options['startup_timeout_s']: break
                elif now-last_command>self.options['watchdog_s'] and now-last_brake>=1/self.cfg['runtime']['send_hz']:
                    self.transmit(STOP);self.report['watchdog_brakes']+=1
                    last_brake=now
                    if now-last_command>self.options['idle_exit_s']: break
        finally:
            try:
                if self.active:
                    for _ in range(3):
                        self.transmit(STOP);time.sleep(.02)
            finally:
                self.receiver.close();self.sender.close()
        return self.report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--session',required=True)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--client-host')
    parser.add_argument('--client-port',type=int)
    args=parser.parse_args()
    if len(args.session)!=32 or any(c not in '0123456789abcdef' for c in args.session):
        parser.error('session must be a 128-bit hexadecimal identifier')
    if bool(args.client_host)!=(args.client_port is not None):
        parser.error('client-host and client-port must be provided together')
    bootstrap=None if args.client_host is None else (args.client_host,args.client_port)
    bridge=ControlBridge(load_config(args.config),args.session,bootstrap)
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        args.report.write_text(json.dumps(bridge.report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
