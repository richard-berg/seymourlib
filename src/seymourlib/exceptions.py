class SeymourError(Exception):
    pass


class SeymourConnectionError(SeymourError):
    pass


class SeymourProtocolError(SeymourError):
    pass


class SeymourTransportError(Exception):
    pass
