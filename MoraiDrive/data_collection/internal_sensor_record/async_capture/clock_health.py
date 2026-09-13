"""Observe Linux clocks without changing system time or NTP configuration."""
import ctypes
import time


def snapshot():
    # Bracket the other readings so each new recording has an explicit mapping
    # uncertainty. CLOCK_MONOTONIC remains the existing receive causal domain.
    before=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    mono=time.monotonic_ns()
    wall=time.time_ns()
    after=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return dict(monotonic_ns=mono,wall_ns=wall,monotonic_raw_ns=(before+after)//2,
                raw_bracket_ns=after-before)


def read_timex():
    if ctypes.sizeof(ctypes.c_long)!=8:
        return dict(available=False,reason='Linux 64-bit timex layout required')
    class Timeval(ctypes.Structure):
        _fields_=[('tv_sec',ctypes.c_long),('tv_usec',ctypes.c_long)]
    class Timex(ctypes.Structure):
        _fields_=[('modes',ctypes.c_uint)]+[(k,ctypes.c_long) for k in ('offset','freq','maxerror','esterror')]+[
            ('status',ctypes.c_int)]+[(k,ctypes.c_long) for k in ('constant','precision','tolerance')]+[
            ('time',Timeval)]+[(k,ctypes.c_long) for k in ('tick','ppsfreq','jitter')]+[
            ('shift',ctypes.c_int)]+[(k,ctypes.c_long) for k in ('stabil','jitcnt','calcnt','errcnt','stbcnt')]+[
            ('tai',ctypes.c_int),('padding',ctypes.c_int*11)]
    fn=ctypes.CDLL(None,use_errno=True).adjtimex
    fn.argtypes=[ctypes.POINTER(Timex)]
    fn.restype=ctypes.c_int
    value=Timex()  # modes=0 is a READ-ONLY query, never ADJ_TICK/ADJ_FREQUENCY.
    result=fn(ctypes.byref(value))
    return dict(available=result>=0,return_status=result,errno=ctypes.get_errno(),modes=value.modes,
                tick_us=value.tick,frequency_ppm=value.freq/65536.,status=value.status,
                system_settings_changed=False)


def evaluate(samples, *, max_rate_error_ppm=5000, max_wall_step_ns=50000000):
    if len(samples)<2:
        raise ValueError('At least two clock samples are required')
    mono=samples[-1]['monotonic_ns']-samples[0]['monotonic_ns']
    raw=samples[-1]['monotonic_raw_ns']-samples[0]['monotonic_raw_ns']
    if min(mono,raw)<=0:
        raise ValueError('Clock probe did not advance')
    ratio=mono/raw
    steps=[]
    for a,b in zip(samples,samples[1:]):
        change=(b['wall_ns']-a['wall_ns'])-(b['monotonic_ns']-a['monotonic_ns'])
        if abs(change)>max_wall_step_ns:
            steps.append(dict(before_monotonic_ns=a['monotonic_ns'],after_monotonic_ns=b['monotonic_ns'],wall_offset_change_ns=change))
    failures=[]
    if abs(ratio-1)*1e6>max_rate_error_ppm:
        failures.append('monotonic_rate_differs_from_raw')
    if steps:
        failures.append('wall_clock_step')
    return dict(passed=not failures,failures=failures,monotonic_per_raw_second=ratio,
                rate_error_ppm=(ratio-1)*1e6,wall_steps=steps,
                max_rate_error_ppm=max_rate_error_ppm,max_wall_step_ns=max_wall_step_ns,
                scope='Gross local clock health; does not calibrate sensor exposure clocks')


def probe(seconds=2.):
    if not .1<=seconds<=10:
        raise ValueError('Clock probe must be bounded to 0.1..10 seconds')
    samples=[snapshot()]
    while (samples[-1]['monotonic_raw_ns']-samples[0]['monotonic_raw_ns'])/1e9<seconds:
        time.sleep(.02)
        samples.append(snapshot())
    return dict(evaluate(samples),samples=samples,timex=read_timex())
