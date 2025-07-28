def is_state_dict_consistent(state_dict: dict):
    """Checks if the given state dictionary (generated via env.get_state_dict()) is consistent where each actor/articulation has the same batch dimension"""
    batch_size = None
    for name in ["actors", "articulations"]:
        if name in state_dict:
            for k, v in state_dict[name].items():
                if batch_size is None:
                    batch_size = v.shape[0]
                else:
                    if v.shape[0] != batch_size:
                        return False
    return True