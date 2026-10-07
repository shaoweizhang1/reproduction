"""Forked shards for the accuracy passes; no upstream counterpart.

Workers refuse to ground new terms, so a forked child never calls into Prolog.
A GPU model is copied to the CPU first: torch cannot re-initialise CUDA in a
forked child. Shards are strided because item cost varies.
"""

import copy
import multiprocessing

import torch


def _refuse(terms):
    raise RuntimeError(
        "accuracy worker needed to ground %s; the model was not fully grounded "
        "before forking" % [str(t) for t in terms]
    )


def cpu_copy(model):
    clone = copy.copy(model)
    clone.neural_networks = copy.deepcopy(model.neural_networks)
    clone.neural_networks.to_device(torch.device("cpu"))
    clone.device = torch.device("cpu")
    return clone


def _child(model, work, indices, conn):
    torch.set_num_threads(1)
    model.trees.ground_all = _refuse
    try:
        with torch.no_grad():
            payload = work(model, indices)
        # Plain Python on the way out: a torch tensor would go through torch's
        # resource_sharer over the pipe and reset the connection.
        conn.send(("ok", payload))
    except BaseException as exc:
        conn.send(("error", "%s: %s" % (type(exc).__name__, exc)))
    finally:
        conn.close()


def run_sharded(model, n, workers, work):
    workers = max(1, min(workers, n))
    shards = [list(range(k, n, workers)) for k in range(workers)]
    if workers == 1:
        with torch.no_grad():
            return [work(model, shards[0])]

    if model.device is not None and model.device.type == "cuda":
        model = cpu_copy(model)

    ctx = multiprocessing.get_context("fork")
    procs, conns = [], []
    for shard in shards:
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child, args=(model, work, shard, child_conn), daemon=True
        )
        proc.start()
        child_conn.close()
        procs.append(proc)
        conns.append(parent_conn)

    payloads = []
    try:
        for conn in conns:
            status, payload = conn.recv()
            if status != "ok":
                raise RuntimeError("accuracy worker failed - %s" % payload)
            payloads.append(payload)
        for proc in procs:
            proc.join()
        return payloads
    except EOFError as exc:
        raise RuntimeError("accuracy worker exited without returning a result") from exc
    finally:
        for conn in conns:
            conn.close()
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join()
