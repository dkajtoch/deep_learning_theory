def flatten_dict(nested_dict, parent_key="", separator="_"):
    flattened_dict = {}

    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            flattened_sub_dict = flatten_dict(value, new_key, separator)
            flattened_dict.update(flattened_sub_dict)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                list_key = f"{new_key}{separator}{i}"
                if isinstance(item, dict):
                    flattened_sub_dict = flatten_dict(item, list_key, separator)
                    flattened_dict.update(flattened_sub_dict)
                else:
                    flattened_dict[list_key] = item
        else:
            flattened_dict[new_key] = value

    return flattened_dict
