from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary using bracket notation."""
    if dictionary is None:
        return None
    return dictionary.get(key)

@register.filter
def split(value, delimiter=','):
    """Split a string into a list using the specified delimiter."""
    return value.split(delimiter) 

@register.filter
def abs(value):
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0 