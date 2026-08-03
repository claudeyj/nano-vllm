_tp_size = 1
_tp_rank = 0
_ep_size = 1
_ep_rank = 0


def initialize_parallel_state(
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
):
    global _tp_size, _tp_rank, _ep_size, _ep_rank
    _tp_size, _tp_rank = tp_size, tp_rank
    _ep_size, _ep_rank = ep_size, ep_rank


def get_tp_size() -> int:
    return _tp_size


def get_tp_rank() -> int:
    return _tp_rank


def get_ep_size() -> int:
    return _ep_size


def get_ep_rank() -> int:
    return _ep_rank
