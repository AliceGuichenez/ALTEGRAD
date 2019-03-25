import random
import string

def merge_params(params, defaults):
    ''' Merge to dict together.
        If the key is not defined in params it will set its value.
        It nerver overwrite the value of params.
        Thus you can merge default with the params of the users.
    '''
    if params is None:
        return defaults
    for key, value in defaults.items():
        if not(key in params):
            params[key] = value
    return params


def random_id(N = 5):
    ''' Return a random string.
        Its usefull to identify a specific dataset or a generation even if it has the same parameter.'''
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


def filter_dict(d, keys):
    filtered = {}
    for key in keys:
        if key in d:
            filtered[key] = d[key]
    return filtered