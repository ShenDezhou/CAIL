import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    temp = gen_micro_macro_result(data)
    result = []
    for name in ["mip"]:
        result.append(round(temp[name] * 100, 2))

    return json.dumps(result, sort_keys=True)
