# Cache maintenance and warmup end-to-end evidence

Date: 2026-08-14 (Asia/Shanghai)

## Reference-aware collection

State: the 81-image lightweight evaluation database with 162 referenced
thumbnail/embedding files. Three controlled orphan fixtures were added under
the three allowed roots.

| Phase | Scanned | Referenced | Eligible orphan | Deleted | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dry-run, age 0 | 165 | 162 | 3 | 0 | 70 |
| Apply, age 0 | 165 | 162 | 3 | 3 | 70 |

The reported samples were exactly `thumbnails/gc-orphan.tmp`,
`embeddings/gc-orphan.tmp`, and `faces/gc-orphan.tmp`. After apply, a real
`city night photography` search succeeded against the retained 81-vector cache.
No model files were in scan scope.

## OpenCLIP background warmup

- Fresh Uvicorn process, OpenCLIP multilingual, CPU, offline model cache.
- Initial status: `idle`, `loaded=false`, `device=null`.
- POST response: HTTP 202, `loading`.
- Final status: `ready`, `loaded=true`, `device=cpu` after 41,099 ms.
- Repeated POST after ready: HTTP 202 / `ready` in 14 ms.
- Observed states: only `loading` and `ready`; no duplicate load was launched.
- The real TCP service on port 8886 was terminated and no listener remained.

These timings measure model load/probe latency on this CPU environment, not
inference throughput. A cached CUDA environment will have different startup
costs.
