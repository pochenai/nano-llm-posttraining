# Global debug switch.
DEBUG = False
LOAD_CHECKPOINT = True


def dprint(*args, **kwargs):
    """Like print(), but only fires when DEBUG is on."""
    if DEBUG:
        print(*args, **kwargs)
