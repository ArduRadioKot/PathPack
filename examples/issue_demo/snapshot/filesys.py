def write_blob(payload):
    """Save incoming bytes to an object store."""
    data = bytes(payload)
    return len(data)
