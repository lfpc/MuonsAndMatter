


def get_design(d:dict):
    detector = {
        # "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 11, "worldSizeY": 11,
        # "worldSizeZ": 100,
        # "magnets": magnets,
        "type": 3,
        "limits": {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        },
        "magnetic_field": d,
    }
    return detector