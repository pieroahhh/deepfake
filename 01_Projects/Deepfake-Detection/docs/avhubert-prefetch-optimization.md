# AV-HuBERT extraction optimization, 16 November 2026

Enabled `OPENBLAS_NUM_THREADS=1` for the extractor and subsequent automatic jobs.
PyTorch/OpenMP remains at eight threads. OpenBLAS handles small audio filterbank
matrix operations; its former eight-thread setting added scheduling overhead.

A read-only CPU benchmark on 12 real cached windows in the pinned container
measured transform time of 1.10 seconds with the existing BLAS setting versus
0.18 seconds with one BLAS thread. An earlier repetition measured 1.54 versus
0.26 seconds. All prepared audio/video tensors were bit-exact at BLAS thread
counts 1, 2, 4 and the original setting. These are component timings, not a
guaranteed end-to-end speedup.

The extractor now reads and transforms up to four existing cached inputs ahead
on one CPU worker. Model inference and any missing-cache face detection stay on
the main thread. No batch-size, precision, model, label or threshold change was
made. Held-out AV-HuBERT inference uses the same CPU prefetch option when cached
inputs are available; a cold cache still needs media preparation.

Validation: 15 targeted host tests passed for exact inputs, RNG/order preservation,
cache hit/miss behavior, decoder equivalence, and feature recovery. The real
extractor's resume parity check also passed against saved feature vectors after
deployment. The original source was backed up in `source-before-prefetch-v2`.

The extractor paused cleanly with 10,925 successful features and 4,098 recorded
failure decisions. Both were preserved in the verified resume snapshot; the
original interrupted cache is retained. Progress now includes preserved windows
in its processed-window count. Initial speed after resume includes scanning the
previous source clips and will be misleading until new work accumulates.

The audiovisual controller was restarted and the automatic continuation controller
was restarted afterward. `watch.ps1` still shows both. The queue remains:
finish audiovisual features -> verify -> fusion -> validation/export -> held-out
1,000 inference. Keep Docker and Windows running. See `AUTO_NEXT.md` for resume.
