from taxes import compute_tax

def checkout_invoice(items):
    """Calculate the checkout invoice amount for a cart."""
    price = sum(items)
    return price + compute_tax(price)
