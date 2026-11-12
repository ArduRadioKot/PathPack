from auth import validate_cookie

def login(cookie):
    return validate_cookie(cookie)
