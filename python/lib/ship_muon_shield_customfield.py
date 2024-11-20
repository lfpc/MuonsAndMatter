import json
import numpy as np

import pickle
import gzip
from matplotlib.path import Path as polygon_path

def get_field():
    with gzip.open('/home/hep/lprate/projects/roxie_ship/outputs/points.pkl', 'rb') as f:
        field_s = pickle.load(f)
    B = field_s['B']
    B[:, 0] *= -1
    B[:, 1] *= -1
    return [field_s['points'],B]
def filter_fields(points,fields,corners, dZ):
    corners = np.array(corners).reshape(-1, 2)
    polygon = polygon_path(corners)
    inside = polygon.contains_points(points[:,:2])
    inside = np.logical_and(inside, points[:,2] > -dZ, points[:,2] < dZ)
    return [points[inside].tolist(),fields[inside].tolist()]


def CreateArb8(arbName, medium, dZ, corners, magField, field_profile,
               tShield, x_translation, y_translation, z_translation, stepGeo):
    assert stepGeo == False
    corners /= 100
    dZ /= 100
    z_translation /= 100

    if field_profile != 'uniform': magField = filter_fields(magField[0],magField[1],corners, dZ)

    tShield['components'].append({
        'corners' : corners,
        'field_profile' : field_profile, #interpolation type
        'field' : magField,
        'name': arbName,
        'dz' : dZ,
        "z_center" : z_translation,
    })

# fields should be 4x3 np array
def create_magnet(magnetName, medium, tShield,
                  fields,field_profile, dX,
                  dY, dX2, dY2, dZ, middleGap,
                  middleGap2,ratio_yoke, gap,
                  gap2, Z, stepGeo, Ymgap = 5):
    fDesign = 8
    dY += Ymgap
    # Assuming 0.5A / mm ^ 2 and 10000At needed, about 200cm ^ 2 gaps are necessary
    # Current design safely above this.Will consult with MISiS to get a better minimum.
    gap = np.ceil(max(100. / dY, gap))
    gap2 = np.ceil(max(100. / dY2, gap2))
    coil_gap = gap
    coil_gap2 = gap2

    anti_overlap = 0.1 # gap between fields in the corners for mitred joints (Geant goes crazy when
    # they touch each other)


    cornersMainL = np.array([
        middleGap, -(dY + ratio_yoke*dX - anti_overlap), middleGap, dY + ratio_yoke*dX - anti_overlap,
        dX + middleGap, dY - anti_overlap, dX + middleGap,
        -(dY - anti_overlap),
        middleGap2, -(dY2 + dX2*ratio_yoke - anti_overlap), middleGap2, dY2 + dX2*ratio_yoke - anti_overlap,
        dX2 + middleGap2, dY2 - anti_overlap, dX2 + middleGap2,
        -(dY2 - anti_overlap)])

    cornersTL = np.array((middleGap + dX,
                            dY,
                            middleGap,
                            dY + dX*ratio_yoke,
                            dX + ratio_yoke*dX + middleGap + coil_gap,
                            dY + dX*ratio_yoke,
                            dX + middleGap + coil_gap,
                            dY,
                            middleGap2 + dX2,
                            dY2,
                            middleGap2,
                            dY2 + dX2*ratio_yoke,
                            dX2 + ratio_yoke*dX2 + middleGap2 + coil_gap2,
                            dY2 + dX2*ratio_yoke,
                            dX2 + middleGap2 + coil_gap2,
                            dY2))

    cornersMainSideL = np.array((dX + middleGap + gap, -(dY - anti_overlap), dX + middleGap + gap,
                                dY - anti_overlap, dX + ratio_yoke*dX + middleGap + gap, dY + ratio_yoke*dX - anti_overlap,
                                dX + ratio_yoke*dX + middleGap + gap, -(dY + ratio_yoke*dX - anti_overlap), dX2 + middleGap2 + gap2,
                                -(dY2 - anti_overlap), dX2 + middleGap2 + gap2, dY2 - anti_overlap,
                                dX2 + ratio_yoke*dX2 + middleGap2 + gap2, dY2 + ratio_yoke*dX2 - anti_overlap, dX2 + ratio_yoke*dX2 + middleGap2 + gap2,
                                -(dY2 + ratio_yoke*dX2 - anti_overlap)))

    cornersMainR = np.zeros(16, np.float64)
    cornersCLBA = np.zeros(16, np.float64)
    cornersMainSideR = np.zeros(16, np.float64)
    cornersCLTA = np.zeros(16, np.float64)
    cornersCRBA = np.zeros(16, np.float64)
    cornersCRTA = np.zeros(16, np.float64)

    cornersTR = np.zeros(16, np.float64)
    cornersBL = np.zeros(16, np.float64)
    cornersBR = np.zeros(16, np.float64)


    # Use symmetries to define remaining magnets
    for i in range(16):
        cornersMainR[i] = -cornersMainL[i]
        cornersMainSideR[i] = -cornersMainSideL[i]
        cornersCRTA[i] = -cornersCLBA[i]
        cornersBR[i] = -cornersTL[i]

    # Need to change order as corners need to be defined clockwise
    for i in range(8):
        j = (11 - i) % 8
        cornersCLTA[2 * j] = cornersCLBA[2 * i]
        cornersCLTA[2 * j + 1] = -cornersCLBA[2 * i + 1]
        cornersTR[2 * j] = -cornersTL[2 * i]
        cornersTR[2 * j + 1] = cornersTL[2 * i + 1]

    for i in range(16):
        cornersCRBA[i] = -cornersCLTA[i]
        cornersBL[i] = -cornersTR[i]

    str1L = "_MiddleMagL"
    str1R = "_MiddleMagR"
    str2 = "_MagRetL"
    str3 = "_MagRetR"
    str4 = "_MagCLB"
    str5 = "_MagCLT"
    str6 = "_MagCRT"
    str7 = "_MagCRB"
    str8 = "_MagTopLeft"
    str9 = "_MagTopRight"
    str10 = "_MagBotLeft"
    str11 = "_MagBotRight"

    stepGeo = False

    theMagnet = {
        'components' : []
    }

    if field_profile == 'uniform':
        CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, fields[0], field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, fields[0], field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, fields[1], field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, fields[1], field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str8, medium, dZ, cornersTL, fields[3], field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str9, medium, dZ, cornersTR, fields[2], field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str10, medium, dZ, cornersBL, fields[2], field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str11, medium, dZ, cornersBR, fields[3], field_profile, theMagnet, 0, 0, Z, stepGeo)

    else:
        CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str8, medium, dZ, cornersTL, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str9, medium, dZ, cornersTR, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str10, medium, dZ, cornersBL, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str11, medium, dZ, cornersBR, fields, field_profile, theMagnet, 0, 0, Z, stepGeo)

    theMagnet['dz'] = dZ/100
    theMagnet['z_center'] = Z/100

    tShield['magnets'].append(theMagnet)


def design_muon_shield(params,fSC_mag = True, use_simulated_fields = False):
    n_magnets = 9
    cm = 1
    mm = 0.1 * cm
    m = 100 * cm
    tesla = 1
    fField = 1.7
    SC_field = 5.1

    magnetName = ["MagnAbsorb1", "MagnAbsorb2", "Magn1", "Magn2", "Magn3", "Magn4", "Magn5", "Magn6", "Magn7"]

    fieldDirection = ["up", "up", "up", "up", "up", "down", "down", "down", "down"]

    zgap = 10 * cm

    LE = 7 * m
    dZ0 = 1 * m
    dZ1 = params[0]#0.4 * m
    dZ2 = params[1] #2.31 * m
    dZ3 = params[2]
    dZ4 = params[3]
    dZ5 = params[4]
    dZ6 = params[5]
    dZ7 = params[6]
    dZ8 = params[7]
    fMuonShieldLength = 2 * (dZ1 + dZ2 + dZ3 + dZ4 + dZ5 + dZ6 + dZ7 + dZ8) + LE


    dXIn = np.zeros(n_magnets)
    dXOut = np.zeros(n_magnets)
    gapIn = np.zeros(n_magnets)
    dYIn = np.zeros(n_magnets)
    dYOut = np.zeros(n_magnets)
    gapOut = np.zeros(n_magnets)
    dZf = np.zeros(n_magnets)
    ratio_yokes = np.ones(n_magnets)

    Z = np.zeros(n_magnets)
    midGapIn= np.zeros(n_magnets)
    midGapOut= np.zeros(n_magnets)
    HmainSideMagIn= np.zeros(n_magnets)
    HmainSideMagOut= np.zeros(n_magnets)

    offset = 7

    # dXIn[0] = 0.4 * m
    # dXOut[0] = 0.40 * m
    # dYIn[0] = 1.5 * m
    # dYOut[0] = 1.5 * m
    # gapIn[0] = 0.1 * mm
    # gapOut[0] = 0.1 * mm
    # ratio_yokes[0] = 1
    # dXIn[1] = 0.5 * m
    # dXOut[1] = 0.5 * m
    # dYIn[1] = 1.3 * m
    # dYOut[1] = 1.3 * m
    # gapIn[1] = 0.02 * m
    # gapOut[1] = 0.02 * m
    # ratio_yokes[1] = 1


    offset = 7

    for i in range(n_magnets-1): #range(2,n_magnets-1)
        dXIn[i] = params[offset + i * 7 + 1]
        dXOut[i] = params[offset + i * 7 + 2]
        dYIn[i] = params[offset + i * 7 + 3]
        dYOut[i] = params[offset + i * 7 + 4]
        gapIn[i] = params[offset + i * 7 + 5]
        gapOut[i] = params[offset + i * 7 + 6]
        ratio_yokes[i] = params[offset + i * 7 + 7]

    XXX = -25 * m - fMuonShieldLength / 2. # TODO: This needs to be checked
    zEndOfAbsorb = XXX - fMuonShieldLength / 2.

    dZf[0] = dZ1 - zgap / 2
    Z[0] = zEndOfAbsorb + dZf[0] + zgap
    dZf[1] = dZ2 - zgap / 2
    Z[1] = Z[0] + dZf[0] + dZf[1] + zgap
    dZf[2] = dZ3 - zgap / 2
    Z[2] = Z[1] + dZf[1] + dZf[2] + 2 * zgap
    dZf[3] = dZ4 - zgap / 2
    Z[3] = Z[2] + dZf[2] + dZf[3] + zgap
    dZf[4] = dZ5 - zgap / 2
    Z[4] = Z[3] + dZf[3] + dZf[4] + zgap
    dZf[5] = dZ6 - zgap / 2
    Z[5] = Z[4] + dZf[4] + dZf[5] + zgap
    dZf[6] = dZ7 - zgap / 2
    Z[6] = Z[5] + dZf[5] + dZf[6] + zgap
    dZf[7] = dZ8 - zgap / 2
    Z[7] = Z[6] + dZf[6] + dZf[7] + zgap

    dXIn[8] = dXOut[7]
    dYIn[8] = dYOut[7]
    dXOut[8] = dXIn[8]
    dYOut[8] = dYIn[8]
    gapIn[8] = gapOut[7]
    gapOut[8] = gapIn[8]
    dZf[8] = 0.1 * m
    Z[8] = Z[7] + dZf[7] + dZf[8]

    for i in range(n_magnets):
        midGapIn[i] = 0.
        midGapOut[i] = 0.
        HmainSideMagIn[i] = dYIn[i] / 2
        HmainSideMagOut[i] = dYOut[i] / 2

    mField = 1.6 * tesla
    fieldsAbsorber = [
        [0., mField, 0.],
        [0., -mField, 0.],
        [-mField, 0., 0.],
        [mField, 0., 0.]
    ]

    nM = 1

    tShield = {
        'magnets':[]
    }
    create_magnet(magnetName[nM], "G4_Fe", tShield, fieldsAbsorber, 'uniform', dXIn[nM], dYIn[nM], dXOut[nM],
                 dYOut[nM], dZf[nM], midGapIn[nM], midGapOut[nM], ratio_yokes[nM],
                 gapIn[nM], gapOut[nM], Z[nM], True, Ymgap=0.0)
    if use_simulated_fields: field_map = get_field()
    for nM in range(2, n_magnets):
        if (dZf[nM] < 1e-5 or nM == 4) and fSC_mag:
            continue
        if nM == 3 and fSC_mag:
            Ymgap = 5
            ironField_s = SC_field * tesla #5.1
        else:
            Ymgap = 0
            ironField_s = fField * tesla #1.7

        if use_simulated_fields and nM ==3 and fSC_mag:
            field_profile = 'nearest'
            fields_s = field_map#get_field()
        
        else:
            field_profile = 'uniform'
            if fieldDirection[nM] == "down":
                ironField_s = -ironField_s
            magFieldIron_s = [0., ironField_s, 0.]
            RetField_s = [0., -ironField_s/ratio_yokes[nM], 0.]
            ConRField_s = [-ironField_s/ratio_yokes[nM], 0., 0.]
            ConLField_s = [ironField_s/ratio_yokes[nM], 0., 0.]
            fields_s = [magFieldIron_s, RetField_s, ConRField_s, ConLField_s]

        create_magnet(magnetName[nM], "G4_Fe", tShield, fields_s, field_profile, dXIn[nM], dYIn[nM], dXOut[nM],
                  dYOut[nM], dZf[nM], midGapIn[nM], midGapOut[nM],ratio_yokes[nM],
                  gapIn[nM], gapOut[nM], Z[nM], nM == 8,Ymgap=Ymgap)
    return tShield





def get_design_from_params(params, z_bias=50., force_remove_magnetic_field=False, fSC_mag:bool = True, use_simulated_fields = False):
    # nMagnets 9

    shield = design_muon_shield(params, fSC_mag, use_simulated_fields = use_simulated_fields)
    # print(shield)

    magnets_2 = []

    max_z = None
    for mag in shield['magnets']:
        mag['z_center'] = mag['z_center'] + z_bias
        components_2 = mag['components']
        components_2 = [{'corners': (np.array(x['corners'])).tolist(),
                         'field_profile': x['field_profile'] if (not force_remove_magnetic_field) else 'uniform',
                         'field': x['field'] if (not force_remove_magnetic_field) else (0., 0., 0.)} for x
                        in components_2]
        mag['components'] = components_2
        mag['material'] = 'G4_Fe'
        mag['fieldX'] = 0.
        mag['fieldY'] = 0.
        mag['fieldZ'] = 0.
        magnets_2.append(mag)

        new_mz = mag['dz'] + mag['z_center'] + 0.05#limit in 31.5
        if max_z is None or new_mz > max_z:
            max_z = new_mz

    shield['magnets'] = magnets_2

    # print(shield)
    shield.update({
        "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 11, "worldSizeY": 11,
        "worldSizeZ": 180,
        "type" : 1,
        "limits" : {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        },

        "sensitive_film": {
            "z_center" : new_mz,
            "dz" : 0.01,
            "dx": 3,
            "dy": 3,
        }
    })


    return shield