def is_in_notebook():
    try:
        from IPython import get_ipython

        ipython = get_ipython()

        if ipython and "IPKernelApp" in ipython.config:  # pragma: no cover
            return True

    except ImportError:
        return False
    return False
