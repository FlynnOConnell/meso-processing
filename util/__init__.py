import re


def parse_value(value_str):
    if value_str.startswith("'") and value_str.endswith("'"):
        return value_str[1:-1]
    if value_str == 'true':
        return True
    if value_str == 'false':
        return False
    if value_str == 'NaN':
        return float('nan')
    if value_str == 'Inf':
        return float('inf')
    if re.match(r'^\d+(\.\d+)?$', value_str):
        return float(value_str) if '.' in value_str else int(value_str)
    if re.match(r'^\[(.*)\]$', value_str):
        return [parse_value(v.strip()) for v in value_str[1:-1].split()]
    return value_str


def parse_key_value(parse_line):
    key_str, value_str = parse_line.split(' = ', 1)
    return key_str, parse_value(value_str)
