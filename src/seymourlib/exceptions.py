class SeymourError(Exception):
    pass


class SeymourConnectionError(SeymourError):
    pass


class SeymourClientClosedError(SeymourConnectionError):
    """The client was closed, and won't connect again."""


class SeymourProtocolError(SeymourError):
    pass


class SeymourTransportError(SeymourError):
    pass
