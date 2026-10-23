"""Build aligned ENU copies of recordings without editing the source CSVs."""
import csv
import argparse
import hashlib
import io
import json
import math

from .section import JammingSection, ROOT, ROUTE_DIRECTORY

from ...config.paths import PREPARED_DIRECTORY_NAME

PREPARED_DIRECTORY = PREPARED_DIRECTORY_NAME


def clip_rows(section, rows, end_forward):
    """Cut at the first crossing; interpolate the last sample onto the line."""
    def finish(clipped):
        # The shared loader removes <=2 cm forward steps. Preserve the exact
        # common endpoint by dropping a nearly coincident penultimate sample.
        while len(clipped) > 2 and end_forward-section.to_local(
                (float(clipped[-2]['x']),float(clipped[-2]['y'])))[1] <= .02000001:
            del clipped[-2]
        return clipped

    previous_forward = None
    for index, row in enumerate(rows):
        forward = section.to_local((float(row['x']), float(row['y'])))[1]
        if forward >= end_forward-1e-9:
            if index == 0:
                raise ValueError("Recording starts beyond the common end line")
            if abs(forward-end_forward) < 1e-9:
                return finish(rows[:index+1])
            previous = rows[index-1]
            fraction = (end_forward-previous_forward)/(forward-previous_forward)
            last = {key: float(previous[key])+fraction*(float(row[key])-float(previous[key]))
                    for key in row}
            a, b = float(previous['yaw_rad']), float(row['yaw_rad'])
            last['yaw_rad'] = a+fraction*math.atan2(math.sin(b-a), math.cos(b-a))
            if 'heading_deg' in last:
                last['heading_deg'] = math.degrees(last['yaw_rad'])
            return finish(rows[:index]+[last])
        previous_forward = forward
    raise ValueError("Recording does not reach the common end line")


def prepare_library(section, names, requested_end_enu=None):
    """Validate all retained segments before publishing a complete library."""
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide unique recording filenames")
    sources = []
    for name in names:
        path = section.output_dir / name
        if path.parent.resolve() != section.output_dir.resolve() or path.suffix != '.csv':
            raise ValueError("Invalid source recording filename")
        status_file = path.with_suffix('.status.json')
        if status_file.exists():
            error = json.loads(status_file.read_text(encoding='utf-8')).get('error')
            if error and not error.startswith(('Recording ', 'Outside Jamming start line')):
                raise ValueError(name+": recording capture failed: "+error)
        raw = path.read_bytes()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        if len(rows) < 10 or not all(math.isfinite(float(v)) for r in rows for v in r.values()):
            raise ValueError(name+": insufficient or non-finite data")
        progress = [section.to_local((float(r['x']), float(r['y'])))[1] for r in rows]
        sources.append((name, rows, fields, max(progress), hashlib.sha256(raw).hexdigest()))
    available_end = min(source[3] for source in sources)
    requested_forward = available_end
    if requested_end_enu is not None:
        if len(requested_end_enu) != 2 or not all(math.isfinite(v) for v in requested_end_enu):
            raise ValueError('Requested end must be finite ENU XY')
        requested_forward = section.to_local(requested_end_enu)[1]
    end_forward = min(available_end, requested_forward)
    prepared, audit = {}, []
    for name, rows, fields, _, digest in sources:
        clipped = clip_rows(section, rows, end_forward)
        section.validate_recording([tuple(float(r[k]) for k in ('x','y','yaw_rad')) for r in clipped])
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clipped)
        prepared[name] = stream.getvalue()
        end = clipped[-1]
        audit.append(dict(name=name, source_sha256=digest, source_points=len(rows),
                          prepared_points=len(clipped), discarded_tail_points=len(rows)-len(clipped),
                          end_enu=[float(end['x']),float(end['y']),float(end['z'])]))
    output = section.root / PREPARED_DIRECTORY
    output.mkdir(exist_ok=True)
    for name, contents in prepared.items():
        target = output / name
        pending = target.with_suffix('.pending.csv')
        pending.write_text(contents, encoding='utf-8')
        pending.replace(target)
    manifest = dict(routes=list(names), directory=PREPARED_DIRECTORY,
                    source_directory=ROUTE_DIRECTORY, common_end_forward_m=end_forward,
                    requested_end_enu=requested_end_enu,
                    requested_end_forward_m=requested_forward,
                    actual_end_before_requested_m=requested_forward-end_forward,
                    common_end_center_enu=list(section.to_world((0.,end_forward))),
                    forward_unit_enu=list(section.forward), audit=audit)
    pending = section.root / 'routes.pending.json'
    pending.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pending.replace(section.root/'routes.json')
    for name in names:
        status_path = (section.output_dir/name).with_suffix('.status.json')
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        if status.get('error'):
            status['original_recording_error'] = status['error']
        status.update(ready_for_planner=True, error=None, prepared_path=PREPARED_DIRECTORY+'/'+name,
                      common_end_forward_m=end_forward, source_frame='ENU')
        status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end-enu', type=float, nargs=2, metavar=('X','Y'))
    args = parser.parse_args()
    section = JammingSection(ROOT)
    requested = args.end_enu
    if requested is None and (section.root/'routes.json').exists():
        requested = json.loads((section.root/'routes.json').read_text(encoding='utf-8')).get('requested_end_enu')
    result = prepare_library(section, sorted(p.name for p in section.output_dir.glob('*.csv')), requested)
    print('Prepared {} routes at forward={:.3f} m'.format(len(result['routes']),result['common_end_forward_m']))
