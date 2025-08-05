


def get_design():
    detector = {
        # "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 11, "worldSizeY": 11,
        # "worldSizeZ": 100,
        # "magnets": magnets,
        "magnetic_field": [0.0,1.0,0.0],
        "type": 3,
        "store_all": True,
        "limits": {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        }
    }
    return detector