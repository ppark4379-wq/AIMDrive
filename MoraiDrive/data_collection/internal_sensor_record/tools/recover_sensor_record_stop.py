"""Observe and, only if required, stop a MORAI Sensor Record session."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nas_storage import powershell
from ssd_storage import SSD_ROOT, WINDOWS_ROOT, windows_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    if not args.name.replace('_','').isalnum() or len(args.name)>96:
        parser.error('Use a simple unique output name')
    output=(args.output.expanduser().resolve() if args.output
            else SSD_ROOT/'sensor_record_auto_v1'/args.name)
    windows_output=windows_path(output)
    source=(ROOT/'tools/stop_sensor_record.ps1').read_text()
    script=("& {\n"+source+"\n} -SimulatorProcessId %d -OutputDirectory '%s' "
            "-AllowedRoot '%s' -MaxAttempts 3"%
            (args.pid,windows_output,WINDOWS_ROOT))
    print(json.dumps(json.loads(powershell(script,timeout=30)),ensure_ascii=False))


if __name__=='__main__':
    main()
