import numpy as np
import pickle
import gzip
from matplotlib.path import Path as polygon_path
from lib import magnet_simulations
#import sys
#sys.path.append('/home/hep/lprate/projects/MuonsAndMatter/python/lib/reference_designs')
from lib.reference_designs.params import new_parametrization, get_magnet_params, sc_v6
import pandas as pd
from os.path import exists
from time import time

def get_field(resimulate_fields = False,
            params = sc_v6,
            file_name = 'data/outputs/fields.pkl',
            **kwargs_field):
    '''Returns the field map for the given parameters. If from_file is True, the field map is loaded from the file_name.'''
    if resimulate_fields:
        fields = simulate_field(params, file_name = file_name,**kwargs_field)
    elif exists(file_name):
        print('Using field map from file', file_name)
        with open(file_name, 'rb') as f:
            fields = pickle.load(f)
    #fields = [fields['points'],fields['B']]
    return fields

def simulate_field(params,
              Z_init = 0,
              fSC_mag:bool = True,
              z_gap = 0.1,
              field_direction = ['up', 'up', 'up', 'up', 'up', 'down', 'down', 'down', 'down'],
              resol = (0.05,0.05,0.05),
              d_space = ((3., 3., (-1, 30.))), 
              file_name = 'data/outputs/fields.pkl'):
    '''Simulates the magnetic field for the given parameters. If save_fields is True, the fields are saved to data/outputs/fields.pkl'''
    t1 = time()
    all_params = pd.DataFrame()
    Z_pos = 0.
    for i, (mag,idx) in enumerate(new_parametrization.items()):
        mag_params = params[idx]
        if mag in ['?', 'M1']: continue
        elif mag == 'M3'and fSC_mag: 
            Z_pos += 2 * mag_params[0]/100 - z_gap
            continue
        if fSC_mag and mag == 'M2': Ymgap = 0.05; B_goal = 5.1; yoke_type = 'Mag2'
        elif mag == 'HA': Ymgap=0.; B_goal = 1.6; yoke_type = 'Mag1'
        else: Ymgap = 0.; B_goal = 1.7; yoke_type = 'Mag3'
        if field_direction[i] == 'down': B_goal *= -1 
        p = get_magnet_params(mag_params, Ymgap=Ymgap, z_gap=z_gap, B_goal=B_goal, yoke_type=yoke_type)
        p['Z_pos(m)'] = Z_pos
        all_params = pd.concat([all_params, pd.DataFrame([p])], ignore_index=True)
        Z_pos += p['Z_len(m)'] + z_gap
        if mag == 'M2': Z_pos += z_gap
    if file_name is not None: all_params.to_csv('data/magnet_params.csv', index=False)
    all_params = all_params.to_dict(orient='list')
    fields = magnet_simulations.run(all_params, d_space=d_space, resol=resol, apply_symmetry=False)
    fields['points'][:,2] += Z_init/100
    fields['B'] *= -1 #simulation is inverted?
    print('Magnetic field simulation took', time()-t1, 'seconds')
    if file_name is not None:
        with open(file_name, 'wb') as f:
            pickle.dump(fields, f)
            print('Fields saved to', file_name)
    return fields

def filter_fields(points,fields,corners, Z,dZ):
    '''Filters the field_map to only include those inside the magnet. Not being used anymore'''
    corners = np.array(corners).reshape(-1, 2)
    points = np.asarray(points)
    fields = np.asarray(fields)
    corners = np.split(corners,2)
    polygon_1 = polygon_path(corners[0])
    polygon_2 = polygon_path(corners[1])
    inside = np.logical_or(polygon_1.contains_points(points[:,:2]),polygon_2.contains_points(points[:,:2]))
    inside = np.logical_and(inside, (points[:,2] > Z-dZ) & (points[:,2] < Z+dZ))
    if inside.sum() == 0:
        raise ValueError('No points inside the magnet')
    return [points[inside],fields[inside]]

def CreateArb8(arbName, medium, dZ, corners, magField, field_profile,
               tShield, x_translation, y_translation, z_translation, stepGeo):
    assert stepGeo == False
    corners /= 100
    dZ /= 100
    z_translation /= 100
    #if field_profile != 'uniform': 
    #    magField = filter_fields(magField[0],magField[1],corners, z_translation,dZ)
    tShield['components'].append({
        'corners' : corners.tolist(),
        'field_profile' : field_profile, #interpolation type
        'field' : magField,
        'name': arbName,
        'dz' : dZ,
        "z_center" : z_translation,
    })


def CreateCavern(shift = 0):
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
    external_rock = (20,20) #entire space is 40x40

    TCC8_length = 170
    dX_TCC8 = 5
    dY_TCC8 = 3.75
    TCC8_shift = (2.3,1.75,-TCC8_length/2)

    ECN3_length = 100
    dX_ECN3 = 8
    dY_ECN3 = 7.5
    ECN3_shift = (3.5,4, ECN3_length/2)

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


def design_muon_shield(params,fSC_mag = True, use_field_maps = False, field_map_file = None):
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
    n_params = 8

    for i in range(n_magnets-1): #range(2,n_magnets-1)
        dXIn[i] = params[offset + i * n_params + 1]
        dXOut[i] = params[offset + i * n_params + 2]
        dYIn[i] = params[offset + i * n_params + 3]
        dYOut[i] = params[offset + i * n_params + 4]
        gapIn[i] = params[offset + i * n_params + 5]
        gapOut[i] = params[offset + i * n_params + 6]
        ratio_yokes[i] = params[offset + i * n_params + 7]
        midGapIn[i] = params[offset + i * n_params + 8]
        midGapOut[i] = midGapIn[i]

    #XXX = -25 * m - fMuonShieldLength / 2. # TODO: This needs to be checked
    #zEndOfAbsorb = XXX - fMuonShieldLength / 2.

    #dZf[0] = dZ1 - zgap / 2
    #Z[0] = zEndOfAbsorb + dZf[0] + zgap
    dZf[1] = dZ2 - zgap / 2
    Z[1] = dZf[1]#Z[0] + dZf[0] + dZf[1] + zgap
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
        #????
        HmainSideMagIn[i] = dYIn[i] / 2
        HmainSideMagOut[i] = dYOut[i] / 2

    tShield = {
        'dz': fMuonShieldLength / 200,
        'magnets':[],
        'global_field_map': []
    }

    if use_field_maps: 
        resimulate_field = (field_map_file is None) or (not exists(field_map_file))
        d_space = (3., 3., (-1, np.ceil((Z[-1]+dZf[-1]+50)/100)))
        resol = (0.05,0.05,0.05)
        field_map = get_field(resimulate_field,np.asarray(params),Z_init = (Z[1] - dZf[1]), fSC_mag=fSC_mag, 
                              resol = resol, d_space = d_space,
                              field_direction = fieldDirection,file_name=field_map_file)
        field_map = {'B': field_map['B'],
                     'range_x': [0,d_space[0], resol[0]],
                     'range_y': [0,d_space[1], resol[1]],
                     'range_z': [d_space[2][0],d_space[2][1], resol[2]]}

        tShield['global_field_map'] = field_map
    for nM in range(1, n_magnets):
        if (dZf[nM] < 1e-5 or nM == 4) and fSC_mag:
            continue
        if nM == 3 and fSC_mag:
            Ymgap = 5*cm
            ironField_s = SC_field * tesla #5.1
        elif nM == 1:
            Ymgap = 0
            ironField_s = 1.6 * tesla
        else:
            Ymgap = 0
            ironField_s = fField * tesla #1.7

        if use_field_maps and nM != 8:# and nM ==3 and fSC_mag:
            field_profile = 'global'
            fields_s = [[],[]]#field_map
        
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
                  gapIn[nM], gapOut[nM], Z[nM], nM in [1,8],Ymgap=Ymgap) #making the last small magnet uniform field, change this?
    return tShield





def get_design_from_params(params, 
                           fSC_mag:bool = True, 
                           force_remove_magnetic_field = False,
                           use_field_maps = False,
                           field_map_file = None,
                           sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6,'position':0},
                           add_cavern:bool = True):

    shield = design_muon_shield(params, fSC_mag, use_field_maps = use_field_maps, field_map_file = field_map_file)
    for mag in shield['magnets']:
        mag['z_center'] = mag['z_center']
        for x in mag['components']:
            if force_remove_magnetic_field:
                x['field'] = (0.0, 0.0, 0.0)
                x['field_profile'] = 'uniform'
            elif x['field_profile'] not in ['uniform','global']: 
                x['field']['B'] = x['field']['B'].tolist()
        mag['material'] = 'G4_Fe'
        max_z = mag['dz'] + mag['z_center'] + 0.05
    if shield['global_field_map'] != []:
        shield['global_field_map']['B'] = shield['global_field_map']['B'].tolist()
    fairship_shift = shield['dz']+25
    z_transition = (-25-shield['dz']+(2*shield['magnets'][0]['dz']+0.8)+0.24+12)
    if add_cavern: shield["cavern"] = CreateCavern(fairship_shift+z_transition)#CreateCavern(shield['dz']+25) #not perfectly consistent with fairship? 30 cm different

    #shield['magnets'] = magnets_2
    if sensitive_film_params is not None:
        sensitive_film_position = sensitive_film_params['position']
        if isinstance(sensitive_film_position, tuple):
            sensitive_film_position = sensitive_film_position[1] + \
                                        shield['magnets'][sensitive_film_position[0]]['z_center'] + \
                                        shield['magnets'][sensitive_film_position[0]]['dz']
        else: sensitive_film_position += max_z
        max_z = sensitive_film_position

    print('TOTAL LENGTH', max_z, shield['dz']*2-7)
    shield.update({
        "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 41, "worldSizeY": 41,
        "worldSizeZ": 400,#max_z*2 + 30,
        "type" : 1,
        "limits" : {
            "max_step_length": 0.05,#-1,
            "minimum_kinetic_energy": 0.1,#-1
        },
    })
    if sensitive_film_params is not None:
        shield.update({
            "sensitive_film": {
            "z_center" : sensitive_film_position,
            "dz" : sensitive_film_params['dz'],
            "dx": sensitive_film_params['dx'],
            "dy": sensitive_film_params['dy']}})


    return shield


if __name__ == '__main__':
    import json
    import numpy as np
    from lib.ship_muon_shield_customfield import get_design_from_params
    from muon_slabs import initialize
    detector = get_design_from_params(np.array(sc_v6), use_field_maps=True,field_map_file = 'data/outputs/fields.pkl', add_cavern=True)
    t1 = time()
    output_data = initialize(np.random.randint(256), np.random.randint(256), np.random.randint(256), np.random.randint(256), json.dumps(detector))#, np.array([0.,0.]))
    print('Time to initialize', time()-t1)
    t1 = time()
    json.dumps(detector)
    print('Time to JSON dump', time()-t1)