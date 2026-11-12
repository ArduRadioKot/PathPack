def read_session(cookie):
    """Read the stored record and decode its expiration timestamp."""
    record = cookie.split(':')
    return record[-1]
