"""One CPU decoder worker, at most one chunk ahead of the consumer.

Only sampling runs on the worker. Face detection, tracking, crop transforms,
model execution, cache writes and progress remain on the calling thread.
The caller must close the iterator on an early exit (including detector errors).
"""
from concurrent.futures import ThreadPoolExecutor


def decoded_chunks(sampler, path, requests, *, fps, enabled=False):
    """Yield samples in request order; never change sampler arguments or errors."""
    if not enabled:
        for start, duration in requests:
            yield sampler(path, start, duration, fps=fps)
        return
    iterator = iter(requests)
    first = next(iterator, None)
    if first is None:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='avtenet-decode') as pool:
        pending = pool.submit(sampler, path, *first, fps=fps)
        try:
            while pending is not None:
                result = pending.result()
                following = next(iterator, None)
                pending = (pool.submit(sampler, path, *following, fps=fps)
                           if following is not None else None)
                yield result
                del result
        finally:
            if pending is not None:
                pending.cancel()
            # Executor exit waits for an already-running decode before the
            # owner can reuse or close its stateful VideoWindowReader.
