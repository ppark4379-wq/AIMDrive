"""MORAI 55-byte EgoCtrlCmd and local bridge envelope; standard library only."""
import json
import math
import struct

PACKET=struct.Struct('<14si3i3b5f2s')


def checked_command(command):
    values={k:float(command[k]) for k in ('accel','brake','steer')}
    if not all(math.isfinite(x) for x in values.values()):
        raise ValueError('Nonfinite control command')
    if not 0<=values['accel']<=1 or not 0<=values['brake']<=1 or not -1<=values['steer']<=1:
        raise ValueError('Control command outside MORAI normalized range')
    if values['brake']>0 and values['accel']>0:
        raise ValueError('Simultaneous throttle and brake')
    return values


def pack_command(command):
    c=checked_command(command)
    return PACKET.pack(b'#MoraiCtrlCmd$',23,0,0,0,2,4,1,0.,0.,c['accel'],c['brake'],c['steer'],b'\r\n')


def encode_message(session,sequence,kind,command=None):
    value=dict(schema='vldrive-control-bridge-v1',session=session,sequence=sequence,kind=kind)
    if command is not None:
        value['command']=checked_command(command)
    return json.dumps(value,allow_nan=False,separators=(',',':')).encode()


def decode_message(payload,session):
    if len(payload)>2048:
        raise ValueError('Oversize control envelope')
    msg=json.loads(payload)
    if (not isinstance(msg,dict) or msg.get('schema')!='vldrive-control-bridge-v1' or msg.get('session')!=session
            or type(msg.get('sequence')) is not int or msg['sequence']<0
            or msg.get('kind') not in ('hello','command','stop')):
        raise ValueError('Invalid control envelope/session')
    if msg['kind']=='command':
        msg['command']=checked_command(msg['command'])
    return msg
