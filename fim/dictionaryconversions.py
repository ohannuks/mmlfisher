keywords = ['lens', 'source', 'light', 'gw']

def flatten_dictionary( kwargs_params ):
    """
    Flattens a dictionary of parameters by processing keys and values.

    Args:
        kwargs_params (dict): A dictionary where keys are strings and values 
                                can be either dictionaries or lists of dictionaries.

    Returns:
        dict: A flattened dictionary with processed keys and values.

    Raises:
        ValueError: If a key without the 'kwargs_' prefix is not in the allowed keywords.
    """
    flattened_kwargs_params = {}
    for kwarg in kwargs_params:
        # Name is kwarg without the 'kwargs_' prefix
        name = kwarg[7:]
        if name not in keywords:
            raise ValueError(f"Invalid keyword '{name}'. Allowed keywords are {keywords}.")
        # Check if the value is a list of dictionaries or a single dictionary
        if isinstance(kwargs_params[kwarg], list):
            # Flatten the list of dictionaries
            for i, d in enumerate(kwargs_params[kwarg]):
                for k, v in d.items():
                    # Create a new key with the format "name_i_k"
                    new_key = f"{name}_{i}_{k}"
                    flattened_kwargs_params[new_key] = v
        else:
            # If it's not a list, just rename it
            flattened_kwargs_params[name] = kwargs_params[kwarg]
    # Return the flattened dictionary
    return flattened_kwargs_params

def unflatten_dictionary( flattened_kwargs_params ):
    """
    Unflattens a dictionary of parameters by processing keys and values.

    Args:
        flattened_kwargs_params (dict): A flattened dictionary where keys are strings 
                                        and values can be either dictionaries or lists of dictionaries.

    Returns:
        dict: An unflattened dictionary with processed keys and values.
    """
    unflattened_kwargs_params = {}
    for kwarg in flattened_kwargs_params:
        # Split the key into parts
        parts = kwarg.split('_')
        name = parts[0]
        if name not in keywords:
            raise ValueError(f"Invalid keyword '{name}'. Allowed keywords are {keywords}.")
        name = "kwargs_" + name
        # Check if the value is a list of dictionaries or a single dictionary
        if len(parts) > 1:
            # Extract the index and key
            index = int(parts[1])
            key = '_'.join(parts[2:])
            # Create the list if it doesn't exist
            if name not in unflattened_kwargs_params:
                unflattened_kwargs_params[name] = [{} for _ in range(index + 1)]
            # Assign the value to the correct index and key
            unflattened_kwargs_params[name][index][key] = flattened_kwargs_params[kwarg]
        else:
            # If it's not a list, just rename it
            unflattened_kwargs_params[name] = flattened_kwargs_params[kwarg]
    # Return the unflattened dictionary
    return unflattened_kwargs_params
