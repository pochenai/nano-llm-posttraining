# Global debug switch.
DEBUG = True


def dprint(*args, **kwargs):
    """Like print(), but only fires when DEBUG is on."""
    if DEBUG:
        print(*args, **kwargs)
