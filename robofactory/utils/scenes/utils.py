def merge_dicts(base, custom):
    """Recursively merge dictionaries, with custom dict taking precedence."""
    for key, value in custom.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base