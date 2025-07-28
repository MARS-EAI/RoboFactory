import functools

def nested_dict_map(f, x):
    """
    Map f over all leaf of nested dict x
    """

    if not isinstance(x, dict):
        return f(x)
    y = dict()
    for key, value in x.items():
        y[key] = nested_dict_map(f, value)
    return y

def nested_dict_reduce(f, x):
    """
    Map f over all values of nested dict x, and reduce to a single value
    """
    if not isinstance(x, dict):
        return x

    reduced_values = list()
    for value in x.values():
        reduced_values.append(nested_dict_reduce(f, value))
    y = functools.reduce(f, reduced_values)
    return y


def nested_dict_check(f, x):
    bool_dict = nested_dict_map(f, x)
    result = nested_dict_reduce(lambda x, y: x and y, bool_dict)
    return result


from robofactory import DIR_MAP

def nested_yaml_map(f, x):
    """
    Map f over all leaf of nested yaml dict x
    """
    if isinstance(x, dict):
        y = dict()
        for key, value in x.items():
            y[key] = nested_yaml_map(f, value)
        return y
    elif isinstance(x, list):
        y = list()
        for value in x:
            y.append(nested_yaml_map(f, value))
        return y
    else:
        return f(x)

def replace_dir(x:str):
    if isinstance(x, str):
        for key in DIR_MAP.keys():
            if key in x:
                x = x.replace(key, DIR_MAP[key])
    return x