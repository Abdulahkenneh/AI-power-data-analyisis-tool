# your_app/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def splitlines(value):
    return value.splitlines()  # Split the string by newlines
