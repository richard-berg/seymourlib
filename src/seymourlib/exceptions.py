class SeymourError(Exception):
    pass


class SeymourConnectionError(SeymourError):
    pass


class SeymourTimeoutError(SeymourConnectionError):
    """The connection is open, but the controller didn't answer.

    Unlike other connection errors, the bridge was reachable: the controller
    may be off or disconnected, the bridge's serial settings may be wrong, or
    the bridge may still hold an old connection.
    """


class SeymourClientClosedError(SeymourConnectionError):
    """The client was closed, and won't connect again."""


class SeymourProtocolError(SeymourError):
    pass


class SeymourTransportError(SeymourError):
    pass
