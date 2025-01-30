import numpy as np
import pickle
import gzip
from matplotlib.path import Path as polygon_path
from lib import magnet_simulations
#import sys
#sys.path.append('/home/hep/lprate/projects/MuonsAndMatter/python/lib/reference_designs')
from lib.reference_designs.params import new_parametrization, get_magnet_params, sc_v6, optimal_oliver
import pandas as pd
from os.path import exists
from time import time
from muon_slabs import initialize
import json

def get_field(resimulate_fields = False,
            params = sc_v6,
            file_name = 'data/outputs/fields.pkl',
            only_grid_params = False,
            **kwargs_field):
    '''Returns the field map for the given parameters. If from_file is True, the field map is loaded from the file_name.'''
    if resimulate_fields:
        fields = simulate_field(params, file_name = file_name,**kwargs_field)['B']
    elif exists(file_name):
        print('Using field map from file', file_name)
        with open(file_name, 'rb') as f:
            if only_grid_params: 
                fields = pickle.load(f)['B']
            else:
                fields = pickle.load(f)
    if only_grid_params: 
        d_space = kwargs_field['d_space']
        resol = kwargs_field['resol']
        fields= {'B': fields.astype(np.float32),
                'range_x': [0,d_space[0], resol[0]],
                'range_y': [0,d_space[1], resol[1]],
                'range_z': [d_space[2][0],d_space[2][1], resol[2]]}
    return fields

def simulate_field(params,
              Z_init = 0,
              fSC_mag:bool = True,
              z_gap = 0.1,
              field_direction = [ 'up', 'up', 'up', 'up', 'down', 'down', 'down', 'down'],
              resol = (0.05,0.05,0.05),
              d_space = ((4., 4., (-1, 30.))), 
              file_name = 'data/outputs/fields.pkl',
              cores = 1):
    '''Simulates the magnetic field for the given parameters. If save_fields is True, the fields are saved to data/outputs/fields.pkl'''
    t1 = time()
    all_params = pd.DataFrame()
    Z_pos = 0.
    for i, (mag,idx) in enumerate(new_parametrization.items()):
        mag_params = params[idx]
        if mag == 'HA': Ymgap=0.; B_goal = 1.6; yoke_type = 'Mag1'
        elif mag in ['M1', 'M2', 'M3']: Ymgap = 0.; B_goal = 1.7; yoke_type = 'Mag1'
        else: Ymgap = 0.; B_goal = 1.7; yoke_type = 'Mag3'
        if fSC_mag:
            if mag == 'M1': continue
            elif mag == 'M3':
                Z_pos += 2 * mag_params[0]/100 - z_gap
                continue
            elif mag == 'M2': Ymgap = 0.05; B_goal = 5.1; yoke_type = 'Mag2'
        if field_direction[i] == 'up': B_goal *= -1 
        p = get_magnet_params(mag_params, Ymgap=Ymgap, z_gap=z_gap, B_goal=B_goal, yoke_type=yoke_type, resol = resol)
        p['Z_pos(m)'] = Z_pos
        all_params = pd.concat([all_params, pd.DataFrame([p])], ignore_index=True)
        Z_pos += p['Z_len(m)'] + z_gap
        if mag == 'M2': Z_pos += z_gap
    if file_name is not None: all_params.to_csv('data/magnet_params.csv', index=False)
    all_params = all_params.to_dict(orient='list')
    fields = magnet_simulations.run(all_params, d_space=d_space, resol=resol, apply_symmetry=False, cores=cores)
    fields['points'][:,2] += Z_init/100
    print('Magnetic field simulation took', time()-t1, 'seconds')
    if file_name is not None:
        with open(file_name, 'wb') as f:
            pickle.dump(fields, f)
            print('Fields saved to', file_name)
    return fields

def filter_fields(points, fields, corners, Z, dZ):
    """
    Filters the field map to only include those inside the 3D figure using Delaunay triangulation.
    
    Args:
        points (array-like): Nx3 array of points to be tested.
        fields (array-like): Corresponding field values for the points.
        corners (array-like): Flat array of shape (16,) representing the corners of two polygons.
                              The first 8 elements correspond to the polygon at Z-dZ,
                              and the last 8 to the polygon at Z+dZ.
        Z (float): Central Z-coordinate to define the height range.
        dZ (float): Half-height of the volume to define the Z range.
        
    Returns:
        list: Filtered points and fields inside the 3D figure.
    """
    from scipy.spatial import Delaunay
    points = np.asarray(points)
    fields = np.asarray(fields)
    corners = np.asarray(corners).reshape(2, 4, 2) 
    lower_polygon = np.hstack([corners[0], np.full((4, 1), Z - dZ)]) 
    upper_polygon = np.hstack([corners[1], np.full((4, 1), Z + dZ)]) 
    volume_corners = np.vstack([lower_polygon, upper_polygon])
    hull = Delaunay(volume_corners)
    inside_hull = hull.find_simplex(points) >= 0
    inside = inside_hull
    if inside.sum() == 0:
        raise ValueError('No points inside the magnet')
    return [points[inside], fields[inside]]

def CreateArb8(arbName, medium, dZ, corners, magField, field_profile,
               tShield, x_translation, y_translation, z_translation, stepGeo):
    assert stepGeo == False
    corners /= 100
    dZ /= 100
    z_translation /= 100

    tShield['components'].append({
        'corners' : corners.tolist(),
        'field_profile' : field_profile,
        'field' : magField,
        'name': arbName,
        'dz' : dZ,
        "z_center" : z_translation,
    })

def constraints_cavern_intersection(corners, dZ, z_translation, cavern_transition):
    def get_cavern_bounds(z):
        if z < cavern_transition:
            y_min = -1.7
            y_max = 5.8
            x_min = -3.56
            x_max = 6.43
        else:
            y_min = -3.36
            y_max = 10
            x_min = -4.56
            x_max = 11.43
        return x_min, x_max, y_min, y_max
    wall_gap = 0.01
    corners = corners.reshape(-1, 2)
    x_min, x_max, y_min, y_max = get_cavern_bounds(z_translation-dZ)
    corners[:8, 0] = np.clip(corners[:8, 0], x_min + wall_gap, x_max - wall_gap)
    corners[:8, 1] = np.clip(corners[:8, 1], y_min + wall_gap, y_max - wall_gap)
    x_min, x_max, y_min, y_max = get_cavern_bounds(z_translation+dZ)
    corners[8:, 0] = np.clip(corners[8:, 0], x_min + wall_gap, x_max - wall_gap)
    corners[8:, 1] = np.clip(corners[8:, 1], y_min + wall_gap, y_max - wall_gap)
    return corners.flatten()

def CreateTarget(z_start:float):
    target_components = []
    N = 13
    T = 5
    materials = N * ["G4_Mo"] + T * ["G4_W"]
    lengths = [8., 2.5, 2.5, 2.5, 2.5, 
        2.5, 2.5, 2.5, 5.0, 5.0, 
        6.5, 8., 8., 5., 8., 
        10., 20., 35.]
    h20_l = 0.5 / 100 # H20 slit *17 times
    diameter  = 30. / 100  # full length in x and y
    z = z_start
    for i in range(N+T):
        L = lengths[i]/100
        if i!=0:
            target_components.append({
            "radius": diameter / 2,
            "dz": h20_l,
            "z_center": z + h20_l/2,
            "material": "G4_WATER",
            })
            z += h20_l
        target_components.append({
            "radius": diameter / 2,
            "dz": L,
            "z_center": z + L/2,
            "material": materials[i],
        })
        z += L
    return target_components

def CreateCavern(shift = 0, length:float = 90.):
    cavern = []
    def cavern_components(x_translation, 
               y_translation, 
               dX,
               dY,
               external_rock):
        x1 = x_translation - dX
        x2 = x_translation + dX
        y1 = y_translation - dY
        y2 = y_translation + dY
        corners_up = np.tile(np.array([[x2, y2],
               [x1, y2],
               [x1, external_rock[1]],
               [x2, external_rock[1]]]).flatten(), 2).tolist()
        corners_right = np.tile(np.array([[x2, -external_rock[1]],
              [x2, external_rock[1]],
              [external_rock[0], external_rock[1]],
              [external_rock[0], -external_rock[1]]]).flatten(), 2).tolist()
        corners_left = np.tile(np.array([[x1, -external_rock[1]],
             [-external_rock[0], -external_rock[1]],
             [-external_rock[0], external_rock[1]],
             [x1, external_rock[1]]]).flatten(), 2).tolist()
        corners_down = np.tile(np.array([[x1, y1],
             [x2, y1],
             [x2, -external_rock[1]],
             [x1, -external_rock[1]]]).flatten(), 2).tolist()
        return [corners_up,corners_right,corners_left,corners_down]
    external_rock = (15,15) #entire space is 40x40

    TCC8_length = max(length, 170.)
    dX_TCC8 = 5
    dY_TCC8 = 3.75
    TCC8_shift = (1.43,2.05,-TCC8_length/2)

    ECN3_length = max(length, 100.)
    dX_ECN3 = 8
    dY_ECN3 = 7
    ECN3_shift = (3.43,3.64, ECN3_length/2)


    TCC8 = {"material": "G4_CONCRETE",
            "name": 'TCC8',
            "dz": TCC8_length/2,
            "z_center" : TCC8_shift[2]+shift,
            "components" : []}
    TCC8["components"] = cavern_components(*TCC8_shift[:2],dX_TCC8,dY_TCC8,external_rock)
    cavern.append(TCC8)

    ECN3 = {"material": "G4_CONCRETE",
            "name": 'ECN3',
            "dz" : ECN3_length/2,
            "z_center" : ECN3_shift[2]+shift,
            "components" : []}
    ECN3["components"] = cavern_components(*ECN3_shift[:2],dX_ECN3,dY_ECN3,external_rock)
    cavern.append(ECN3)

    return cavern

# fields should be 4x3 np array
def create_magnet(magnetName, medium, tShield,
                  fields,field_profile, dX,
                  dY, dX2, dY2, dZ, middleGap,
                  middleGap2,ratio_yoke, gap,
                  gap2, Z, stepGeo, Ymgap = 5):
    dY += Ymgap
    # Assuming 0.5A / mm ^ 2 and 10000At needed, about 200cm ^ 2 gaps are necessary
    # Current design safely above this.Will consult with MISiS to get a better minimum.
    #gap = np.ceil(max(100. / dY, gap))
    #gap2 = np.ceil(max(100. / dY2, gap2))
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

    cornersTL = np.array((middleGap + dX,dY,
                            middleGap,dY + dX*ratio_yoke,
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
        'components' : [],
        'dz' : dZ/100,
        'z_center' : Z/100,
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
    

    tShield['magnets'].append(theMagnet)

def construct_block(medium, tShield,field_profile, stepGeo):
    #iron block before the magnet
    z_gap = 1.
    dX = 356.#395.
    dY = 170.#340.
    dZ = 45. - z_gap/2
    Z = -45.
    cornersIronBlock = np.array([
        -dX, -dY,
        dX, -dY,
        dX, dY,
        -dX, dY,
        -dX, -dY,
        dX, -dY,
        dX, dY,
        -dX, dY
    ])
    Block = {
        'components' : [],
        'dz' : dZ/100,
        'z_center' : Z/100,
    }
    #cornersIronBlock = contraints_cavern_intersection(cornersIronBlock/100, dZ/100, Z/100, 22-2.345)
    CreateArb8('IronAfterTarget', medium, dZ, cornersIronBlock, [0.,0.,0.], field_profile, Block, 0, 0, Z, stepGeo)
    tShield['magnets'].append(Block)

def design_muon_shield(params,fSC_mag = True, simulate_fields = False, field_map_file = None, cores_field:int = 1):
    extra_magnet = False
    n_magnets = 7 + int(extra_magnet)
    cm = 1
    mm = 0.1 * cm
    m = 100 * cm
    tesla = 1

    magnetName = ["MagnAbsorb2", "Magn1", "Magn2", "Magn3", "Magn4", "Magn5", "Magn6", "Magn7"]

    fieldDirection = ["up", "up", "up", "up", "down", "down", "down", "down"]

    zgap = 10 * cm

    dZ1 = params[0] #2.31 * m
    dZ2 = params[1]
    dZ3 = params[2]
    dZ4 = params[3]
    dZ5 = params[4]
    dZ6 = params[5]
    dZ7 = params[6]
    fMuonShieldLength = 2 * (dZ1 + dZ2 + dZ3 + dZ4 + dZ5 + dZ6 + dZ7) + (7 * zgap / 2) + 0.1


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


    offset = 6
    n_params = 8

    for i in range(n_magnets - int(extra_magnet)):
        dXIn[i] = params[offset + i * n_params + 1]
        dXOut[i] = params[offset + i * n_params + 2]
        dYIn[i] = params[offset + i * n_params + 3]
        dYOut[i] = params[offset + i * n_params + 4]
        gapIn[i] = params[offset + i * n_params + 5]
        gapOut[i] = params[offset + i * n_params + 6]
        ratio_yokes[i] = params[offset + i * n_params + 7]
        midGapIn[i] = params[offset + i * n_params + 8]
        midGapOut[i] = midGapIn[i]

    dZf[0] = dZ1 - zgap / 2
    Z[0] = dZf[0]
    dZf[1] = dZ2 - zgap / 2
    Z[1] = Z[0] + dZf[0] + dZf[1] + 2 * zgap
    dZf[2] = dZ3 - zgap / 2
    Z[2] = Z[1] + dZf[1] + dZf[2] + zgap
    dZf[3] = dZ4 - zgap / 2
    Z[3] = Z[2] + dZf[2] + dZf[3] + zgap
    dZf[4] = dZ5 - zgap / 2
    Z[4] = Z[3] + dZf[3] + dZf[4] + zgap
    dZf[5] = dZ6 - zgap / 2
    Z[5] = Z[4] + dZf[4] + dZf[5] + zgap
    dZf[6] = dZ7 - zgap / 2
    Z[6] = Z[5] + dZf[5] + dZf[6] + zgap

    if extra_magnet:
        dXIn[7] = dXOut[6] #last small magnet
        dYIn[7] = dYOut[6]
        dXOut[7] = dXIn[7]
        dYOut[7] = dYIn[7]
        gapIn[7] = gapOut[6]
        gapOut[7] = gapIn[7]
        dZf[7] = 0.1 * m
        Z[7] = Z[6] + dZf[6] + dZf[7]

    for i in range(n_magnets):
        #????
        HmainSideMagIn[i] = dYIn[i] / 2
        HmainSideMagOut[i] = dYOut[i] / 2

    tShield = {
        'dz': fMuonShieldLength / 200,
        'magnets':[],
        'global_field_map': {'B': np.array([])},
    }



    if field_map_file is not None or simulate_fields: 
        simulate_fields = (not exists(field_map_file)) or simulate_fields
        max_x = max(np.max(dXIn + dXIn * ratio_yokes + gapIn+midGapIn), np.max(dXOut + dXOut * ratio_yokes+gapOut+midGapOut))/100
        max_y = max(np.max(dYIn + dXIn * ratio_yokes), np.max(dYOut + dXOut * ratio_yokes))/100
        d_space = (max_x+0.5, max_y+0.5, (-1, np.ceil((Z[-1]+dZf[-1]+50)/100)))
        resol = (0.02,0.02,0.05)
        field_map = get_field(simulate_fields,np.asarray(params),Z_init = (Z[0] - dZf[0]), fSC_mag=fSC_mag, 
                              resol = resol, d_space = d_space,
                              field_direction = fieldDirection,file_name=field_map_file, only_grid_params=True, cores = min(cores_field,n_magnets))
        tShield['global_field_map'] = field_map

    for nM in range(n_magnets):
        if fSC_mag and (dZf[nM] < 1*mm or nM == 3):
            continue
        if nM == 2 and fSC_mag:
            Ymgap = 5*cm
            ironField_s = 5.1 * tesla
        elif nM == 0:
            Ymgap = 0
            ironField_s = 1.6 * tesla
        else:
            Ymgap = 0
            ironField_s = 1.7 * tesla

        if simulate_fields:
            field_profile = 'global'
            fields_s = [[],[],[]]
        else:
            field_profile = 'uniform'
            if fieldDirection[nM] == "down":
                ironField_s = -ironField_s
            if nM in [4,5,6]: ironField_s *= ratio_yokes[nM]
            magFieldIron_s = [0., ironField_s, 0.]
            RetField_s = [0., -ironField_s/ratio_yokes[nM], 0.]
            ConRField_s = [-ironField_s/ratio_yokes[nM], 0., 0.]
            ConLField_s = [ironField_s/ratio_yokes[nM], 0., 0.]
            fields_s = [magFieldIron_s, RetField_s, ConRField_s, ConLField_s]

        create_magnet(magnetName[nM], "G4_Fe", tShield, fields_s, field_profile, dXIn[nM], dYIn[nM], dXOut[nM],
                  dYOut[nM], dZf[nM], midGapIn[nM], midGapOut[nM],ratio_yokes[nM],
                  gapIn[nM], gapOut[nM], Z[nM], nM in [1,7],Ymgap=Ymgap)
    field_profile = 'global' if simulate_fields else 'uniform'
    construct_block("G4_Fe", tShield, field_profile, False)
    return tShield





def get_design_from_params(params, 
                           fSC_mag:bool = True, 
                           force_remove_magnetic_field = False,
                           simulate_fields = False,
                           field_map_file = None,
                           sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6,'position':82},
                           add_cavern:bool = True,
                           add_target:bool = True,
                           cores_field:int = 1):
    params = np.round(params, 2)
    shield = design_muon_shield(params, fSC_mag, simulate_fields = simulate_fields, field_map_file = field_map_file, cores_field=cores_field)
    shift = -2.345
    cavern_transition = 22+shift #m
    World_dZ = 200 #m
    World_dX = World_dY = 30 if add_cavern else 15
    if add_cavern: shield["cavern"] = CreateCavern(cavern_transition, length = World_dZ)
    if add_target: shield['target'] = CreateTarget(z_start=shift)
    #z_bias = sensitive_film_params['position']/2
    i=0
    max_z = 0
    for mag in shield['magnets']:
        i+=1
        mag['z_center'] = mag['z_center']
        for x in mag['components']:
            if force_remove_magnetic_field:
                x['field'] = (0.0, 0.0, 0.0)
                x['field_profile'] = 'uniform'
            elif x['field_profile'] not in ['uniform','global']: 
                x['field']['B'] = x['field']['B'].tolist()
            #if add_cavern: x['corners'] = contraints_cavern_intersection(np.array(x['corners']), x['dz'], x['z_center'], cavern_transition).tolist()
        mag['material'] = 'G4_Fe'
        if mag['dz'] + mag['z_center'] > max_z:
            max_z = mag['dz'] + mag['z_center']

    #assert(False)
    print('TOTAL LENGTH', max_z, shield['dz']*2)
    shield.update({
     "worldSizeX": World_dX, "worldSizeY": World_dY, "worldSizeZ": World_dZ,
        "type" : 1,
        "limits" : {
            "max_step_length": 0.05,
            "minimum_kinetic_energy": 0.1,
        },
    })
    if sensitive_film_params is not None:
        shield.update({
            "sensitive_film": {
            "z_center" : sensitive_film_params['position'],
            "dz" : sensitive_film_params['dz'],
            "dx": sensitive_film_params['dx'],
            "dy": sensitive_film_params['dy']}})
    return shield

def initialize_geant4(detector, seed = None):
    B = detector['global_field_map'].pop('B').flatten()
    if seed is None: seeds = (np.random.randint(256), np.random.randint(256), np.random.randint(256), np.random.randint(256))
    else: seeds = (seed, seed, seed, seed)
    output_data = initialize(*seeds,json.dumps(detector), np.asarray(B))
    return output_data

if __name__ == '__main__':
    import json
    import numpy as np
    from lib.ship_muon_shield_customfield import get_design_from_params
    from muon_slabs import initialize
    import os
    file_map_file = 'data/outputs/fields_mm.pkl'
    if file_map_file is not None and os.path.exists(file_map_file):
        os.remove(file_map_file)
    t1 = time()
    params = optimal_oliver
    detector = get_design_from_params(params, simulate_fields=True,field_map_file = file_map_file, add_cavern=True, cores_field=7, fSC_mag = False)
    t1_init = time()
    output_data = initialize_geant4(detector)
    print('Time to initialize', time()-t1_init)
    print('TOTAL TIME', time()-t1)
    
    t1 = time()
    json.dumps(detector)
    print('Time to JSON dump', time()-t1)
