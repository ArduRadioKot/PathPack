from .storage import get_user


def login(user_id: int) -> str:
    user = get_user(user_id)
    return user["name"]
