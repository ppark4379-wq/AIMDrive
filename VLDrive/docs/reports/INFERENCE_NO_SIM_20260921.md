# Inference without local MORAI ? 2026-09-21

## Measured hardware and method

Intel Core Ultra 7 265K, RTX 5060 Ti 16GB, WSL PyTorch environment, CPU threads=2. Latest epoch-30 model-last.pt, unchanged production models/config, FP32 encoder, BF16 stage 2, batch 1, eleven frames, Gumbel selection, reconstruction disabled. MORAI process absent before and after measurement. GPU background utilization fell from ~91% to 6% before benchmarking.

Six recorded scene anchors distributed across the cached route. Cameras and LiDAR were loaded into RAM before timing. Isolated stage-2 uses actual recorded eleven-frame cache windows. Sequential full timing includes shared CPU preprocessing, host-to-device transfer, one fresh encoder pass, and stage-2 forward with ten cached history frames. Five warmup iterations were excluded. Pipelined replay cycles recorded scenes only to exercise computation; it is not an accuracy/closed-loop test.

Network packet assembly, JPEG decoding, navigation updates, visualization and PID were excluded. This is a participant-PC compute benchmark, not a measured server-to-client LAN experiment. There is no physical 5090 available in this environment.

## Results

- stage2_only: mean 70.914ms, median 70.339ms, p95 78.980ms, samples 48, inverse mean 14.102/s.
- encoder_only: mean 45.113ms, median 47.359ms, p95 51.367ms, samples 30, inverse mean 22.167/s.
- cpu_preprocessing: mean 12.940ms, median 12.568ms, p95 14.543ms, samples 30, inverse mean 77.279/s.
- sequential_cpu_encoder_stage2: mean 132.131ms, median 133.096ms, p95 146.303ms, samples 30, inverse mean 7.568/s.
- pipeline_1: 100 outputs / 20s = 5.00Hz; preprocessing-start-to-output mean 130.15ms; p95 143.80ms; resets/drops/discards 0/0/0.
- pipeline_2: 100 outputs / 20s = 5.00Hz; preprocessing-start-to-output mean 135.05ms; p95 150.42ms; resets/drops/discards 0/0/0.

The replay has 5Hz arrivals after a 3s history warmup. Production nominal history interval is .2s with .06s tolerance; prediction_hz is null, but unchanged fresh-anchor cadence still matters. 7.57Hz sequential processing is a benchmark result, not an absolute hardware maximum. Cached-stage2-only 14.10Hz excludes encoder/preprocessing and is not full driving throughput.

## RTX 5090 conditional estimate

Official NVIDIA comparison: RTX 5090 32GB, 21760 CUDA cores, 1792GB/s bandwidth; 5060 Ti 16GB, 4608 cores, 448GB/s. These are specifications, not VLDrive speed ratios. https://www.nvidia.com/en-us/geforce/graphics-cards/compare/

If the measured approximately 116ms encoder+stage2 portion becomes 2?3 times faster, while the remaining approximately 16ms is unchanged, sequential processing would take 55?74ms (~13?18Hz; rounded planning range 13?19Hz). This is an assumed speedup scenario, not a 5090 measurement or guaranteed forecast. Python/CPU launch overhead within model calls may prevent that scaling. VRAM capacity alone does not double throughput. With the same nominal 5Hz input history, fresh-output rate would remain around 5Hz if all processing fits the interval; reduced latency and additional headroom are the expected benefits. LAN and simulator server delivery remain to be tested.

## Raw measurements

```json
{
  "device": "NVIDIA GeForce RTX 5060 Ti",
  "checkpoint": "/mnt/c/Users/autonav009/Desktop/AIMDrive/VLDrive/outputs/logs/vldrive_20260920_172744_065238/model-last.pt",
  "threads": 2,
  "stage1_precision": "fp32",
  "stage2_precision": "bf16",
  "input_frames": [
    20,
    680,
    1278,
    1826,
    2306,
    2983
  ],
  "limitations": "No network, dashboard, JPEG decoding, navigation or controller; raw images predecoded in RAM. Replayed scenes measure timing, not driving accuracy.",
  "stage2_only": {
    "n": 48,
    "mean_ms": 70.91351349905987,
    "median_ms": 70.33850200241432,
    "p95_ms": 78.98022137233056,
    "hz": 14.101684582491563
  },
  "encoder_only": {
    "n": 30,
    "mean_ms": 45.1127436516496,
    "median_ms": 47.35899297520518,
    "p95_ms": 51.36723415344022,
    "hz": 22.16668548740404
  },
  "sequential_cpu_encoder_stage2": {
    "n": 30,
    "mean_ms": 132.130853269094,
    "median_ms": 133.0959690385498,
    "p95_ms": 146.30316722905263,
    "hz": 7.5682550688099175
  },
  "cpu_preprocessing": {
    "n": 30,
    "mean_ms": 12.94012189997981,
    "median_ms": 12.567703030072153,
    "p95_ms": 14.543125143973155,
    "hz": 77.27902470544426
  },
  "pipeline_1": {
    "outputs": 100,
    "measurement_seconds": 20,
    "hz": 5.0,
    "resets": 0,
    "drops": 0,
    "discarded": 0,
    "stage2_ms": {
      "n": 100,
      "mean_ms": 70.78069435665384,
      "median_ms": 70.35009702667594,
      "p95_ms": 77.54320516833104,
      "hz": 14.12814622813874
    },
    "sensor_to_output_ms": {
      "n": 100,
      "mean_ms": 130.1470735645853,
      "median_ms": 129.36899351188913,
      "p95_ms": 143.79738608258776,
      "hz": 7.683614948927388
    },
    "encoder_ms": {
      "n": 116,
      "mean_ms": 42.57427801374057,
      "median_ms": 41.35995503747836,
      "p95_ms": 51.53840398998,
      "hz": 23.488360734555652
    },
    "preprocessing_ms": {
      "n": 116,
      "mean_ms": 13.595714415463716,
      "median_ms": 12.763984501361847,
      "p95_ms": 17.086260952055454,
      "hz": 73.5525893999806
    }
  },
  "pipeline_2": {
    "outputs": 100,
    "measurement_seconds": 20,
    "hz": 5.0,
    "resets": 0,
    "drops": 0,
    "discarded": 0,
    "stage2_ms": {
      "n": 100,
      "mean_ms": 71.73874005675316,
      "median_ms": 71.12655596574768,
      "p95_ms": 83.35382684599608,
      "hz": 13.939469792874688
    },
    "sensor_to_output_ms": {
      "n": 100,
      "mean_ms": 135.0465079152491,
      "median_ms": 134.26880002953112,
      "p95_ms": 150.41918677743524,
      "hz": 7.404856411596872
    },
    "encoder_ms": {
      "n": 116,
      "mean_ms": 45.66637800649162,
      "median_ms": 45.29021948110312,
      "p95_ms": 53.69071604218334,
      "hz": 21.89794863647489
    },
    "preprocessing_ms": {
      "n": 116,
      "mean_ms": 13.787560795176903,
      "median_ms": 12.890343961771578,
      "p95_ms": 17.63951129396446,
      "hz": 72.52914528216007
    }
  },
  "peak_gpu_gib": 3.31368350982666
}
```
