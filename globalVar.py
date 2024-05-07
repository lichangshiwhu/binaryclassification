

def _init():
    global _global_dict 
    _global_dict = {'Model': None,}

def setValue(key, value):
    _global_dict[key] = value

def getValue(key):
    try:
        return _global_dict[key]
    except:
        print('read ' + key + 'failed\n')
        assert(False)

