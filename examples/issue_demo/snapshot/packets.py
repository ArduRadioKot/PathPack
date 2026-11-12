def encode_packet(message):
    """Serialize transport payload using the wire format."""
    return message.encode('utf8')
