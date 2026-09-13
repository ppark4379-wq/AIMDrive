# Python runtime instability investigation — 2026-09-10

## Confirmed observations

- Active WSL: Ubuntu-20.04, WSL 2.7.8.0, kernel 6.18.33.1-microsoft-standard-WSL2.
- At the controlled checks, Collector, Expert and Sync Runtime were not running. ROS master/roslaunch processes remained. WSL used about 0.8 GiB of its 15 GiB RAM with swap unused.
- `probe_python_runtime.py` uses only Python standard-library modules, creates 65 small points, and performs deterministic projection calculations. It does not import ROS/NumPy, read sensor/map files, connect to MORAI/NAS, or write a dataset.
- Three consecutive runs of this minimal probe on system Python 3.8.10: AttributeError (`SimpleNamespace` has no `enumerate`), PASS, SIGSEGV. Peak RSS was 11,060–11,248 KiB. A preceding run failed its numeric-result assertion.
- Identical minimal probe on the official Windows Python 3.8.10 embeddable build: PASS for 20,000 projections. This controls the Python release number, but not compiler, OS, timing or CPU scheduling.
- Earlier map-based pure-Python checks (also without ROS/NumPy) failed 3/3 at approximately 36 MiB peak RSS: AttributeError, SIGSEGV and TypeError.
- Independent Python 3.8.20 and NumPy 1.24.4 did not eliminate failures in sustained map tests; a SIGFPE was recorded. They were installed only under expert/.runtime for comparison. System Python and NumPy were not upgraded.
- Kernel records include SIGSEGV in rosbridge as well as diagnostic Python processes. Diagnostic-induced crashes must not be mistaken for original collector crashes. These records show user-process crashes, not proof of a WSL VM/kernel crash.
- No OOM-kill record was found in the inspected kernel log. No Windows WHEA-Logger record was returned for the preceding 14 days. Neither absence rules out all resource/hardware problems.
- Host: Core i9-14900KF, Gigabyte Z790 AORUS ELITE, BIOS FG dated 2023-09-12.

## What these results do and do not establish

The current failures can occur without a running data collector and with very small memory use. Dataset capacity, NAS writes, image decoding and collector code are not required for this reproduction. A collector defect cannot by itself explain every observed failure.

They do NOT establish whether collection originally triggered a persistent host/guest problem. No clean pre-collection baseline after reboot has been measured. They do NOT prove a defective CPU, RAM or WSL kernel. A short successful test does not establish stability: both passes and failures occurred in the same environment.

A similar public WSL issue initially blamed kernel 6.18, but the reporter later withdrew that conclusion and reported an i9-13900K hardware issue. This is supporting context, not a diagnosis of this PC:
https://github.com/microsoft/WSL/issues/41019#issuecomment-4928291559
Intel's platform guidance:
https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html

## Next controlled comparison

1. With user authorization, fully stop/restart WSL. This interrupts all WSL workloads.
2. Before starting ROS, MORAI connection, Expert, Runtime or Collector, repeat:
   `python3 -S diagnostics/probe_python_runtime.py`
   from MoraiDrive. Repeat identical runs and retain exit codes/output.
3. If the clean baseline fails, investigate host/WSL stability independently of collection. A PC reboot and properly controlled BIOS/CPU/RAM checks are next; do not infer hardware failure solely from this probe.
4. Only if the baseline is stable, compare staged runs: driving without Collector; Collector with local output; Collector with NAS output. Keep duration, sensors and capture rate fixed; monitor RAM, disk latency, process exit codes and probe results. Do not use missing historical gRPC labels as evidence of an OS fault.
5. Run this same minimal probe on each other affected PC and record CPU, WSL/kernel version and its exact error. Similar symptoms alone do not establish one shared cause.

No driving logic, sensor settings, BIOS or WSL configuration was changed by this investigation. A new diagnostic script and this report were added. Detailed local outputs are under expert/.runtime/minimal_runtime_repro_results.json and collection_cause_repro.json.