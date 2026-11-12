from packets import encode_packet

def send_notification(message):
    """Notify the app user with a packet."""
    return encode_packet(message)
