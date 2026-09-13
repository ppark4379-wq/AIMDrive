"""Lossless native GPS/IMU text validation; not a ROS Header decoder."""
import base64
import math

REQUIRED={
    'gps':{'Latitude','Longitude','Altitude','EastOffset','NorthOffset'},
    'imu':{'TimeStamp.Secs','TimeStamp.nSecs','Orientation X','Orientation Y','Orientation Z','Orientation W',
           'Angular Velocity X','Angular Velocity Y','Angular Velocity Z',
           'Linear Acceleration X','Linear Acceleration Y','Linear Acceleration Z'},
}


def parse_native_text(role,raw):
    text=raw.decode('utf-8-sig');values={}
    for line in text.splitlines():
        if not line.strip():continue
        key,value=line.split(':',1);key=key.strip()
        if key in values:raise ValueError('Duplicate native diagnostic field: '+key)
        values[key]=float(value.strip())
    if not REQUIRED[role]<=set(values) or not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Missing/nonfinite native diagnostic fields: '+role)
    if role=='gps' and not (-90<=values['Latitude']<=90 and -180<=values['Longitude']<=180):
        raise ValueError('Invalid GPS coordinates')
    if role=='imu' and any(values[k]<0 or values[k]!=int(values[k]) for k in ('TimeStamp.Secs','TimeStamp.nSecs')):
        raise ValueError('Invalid native IMU timestamp fields')
    if role=='imu' and abs(sum(values['Orientation '+k]**2 for k in ('X','Y','Z','W'))-1)>.01:
        raise ValueError('Invalid native IMU quaternion')
    return dict(original_text=text,original_bytes_base64=base64.b64encode(raw).decode('ascii'),fields=values)


def imu_clock_candidates(fields):
    """Return hypotheses. Only a multi-frame audit may identify this clock.

    The observed MORAI build writes cumulative ns modulo uint32 in nSecs.
    Never assume that applies to ROS messages or another simulator version.
    """
    secs=int(fields['TimeStamp.Secs']);ns=int(fields['TimeStamp.nSecs'])
    return dict(raw_secs=secs,raw_nsecs=ns,normalized_ros_fields=0<=ns<1000000000,
                ros_style_sum_ns=secs*1000000000+ns,cumulative_nsecs_candidate_ns=ns)


def identify_imu_clock(samples):
    """Compare every raw native text field with independent native tick time.

    A short sequence cannot establish wrap semantics. Require observed values
    above one billion; require crossing 2**32 ns to identify uint32 wrap.
    No raw timestamp is repaired, and this never assigns camera/LiDAR GT.
    """
    high=any(v['raw_nsecs']>=1000000000 for v in samples)
    cumulative=bool(samples and high and all(
        v['raw_nsecs']==v['expected_next_tick_ns'] and
        v['raw_secs']==v['expected_next_tick_ns']//1000000000 for v in samples))
    wraps=bool(samples and high and any(v['expected_next_tick_ns']>=2**32 for v in samples) and all(
        v['raw_nsecs']==v['expected_next_tick_ns']%(2**32) and
        v['raw_secs']==v['expected_next_tick_ns']//1000000000 for v in samples))
    name=('cumulative_nSecs_mod_uint32_equals_next_native_tick' if wraps else
          'cumulative_nSecs_equals_next_native_tick' if cumulative else 'not_identified')
    return dict(observed_format=name,wrap_period_ns=2**32 if wraps else None,
        samples_after_first_uint32_wrap=sum(v['expected_next_tick_ns']>=2**32 for v in samples),
        warning='Raw fields are preserved. Observed native nSecs is cumulative and can wrap at 2**32 ns; '
                'Secs*1e9+nSecs is not a valid clock decoder for this format. This does not apply to ROS Header.')
