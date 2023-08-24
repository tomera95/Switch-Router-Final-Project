KEEP_THRESHOLD = 100


def aifo(quantile, big_c, k, aifo_queue):
    lil_c = len(aifo_queue)
    threshold = (big_c - lil_c) / (big_c * (1 - k))
    if quantile <= threshold:
        aifo_queue.appendleft(quantile)
        return True
    return False


def aifo_with_fifo(quantile, big_c, k, aifo_queue, keep_fifo):
    admissioned = aifo(quantile, big_c, k, aifo_queue)
    if not admissioned:
        if len(keep_fifo) >= KEEP_THRESHOLD:
            keep_fifo.pop()
        keep_fifo.appendleft(quantile)


def sp_aifo(quantile, big_c, k, queue_map, win_size):
    aifo_quque = queue_map[quantile * win_size]
    aifo(quantile, big_c, k, aifo_quque)
