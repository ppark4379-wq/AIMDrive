#!/usr/bin/env python3
"""Analyze a standalone recording/replay directory; no simulator connection."""
import argparse
import csv
import json
from pathlib import Path
from localization_metrics import analyze


def write_analysis(root, output=None):
    output = Path(output) if output else Path(root)/'analysis'
    output.mkdir(parents=True,exist_ok=True)
    result = analyze(root)
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'samples.csv').open('w',newline='',encoding='utf-8') as stream:
        fields = ['gps_received_ns','hold_s','distance_m','error_2d_m','longitudinal_m','lateral_m']
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(result['samples'])
    return result,output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    result,output=write_analysis(args.recording,args.output_dir)
    print(json.dumps(dict(comparisons=result['unique_gps_comparisons'],metrics=result['metrics'],
                          assessment=result['assessment'],output=str(output.resolve())),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
