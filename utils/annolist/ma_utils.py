# TensorBoxPy3 https://github.com/SMH17/TensorBoxPy3

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
