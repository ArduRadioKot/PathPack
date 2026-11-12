from store import read_session

def login_with_cookie(cookie):
    """Validate an incoming login session from a cookie."""
    return read_session(cookie)
