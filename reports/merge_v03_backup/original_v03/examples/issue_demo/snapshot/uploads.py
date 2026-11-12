from filesys import write_blob

def upload_image(payload):
    """Serve the user image upload request."""
    return write_blob(payload)
