from os.path import exists, join
from os import getenv, environ
environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pickle
from lib import magnet_simulations
from time import time
from muon_slabs import initialize
import json
import h5py

from snoopy import RacetrackCoil, compute_prices, get_NI
from pandas import DataFrame

RESOL_DEF = magnet_simulations.RESOL_DEF
MATERIALS_DIR = join(getenv('PROJECTS_DIR'),'MuonsAndMatter/data/materials')
Z_GAP = 10 # in cm
SC_Ymgap = magnet_simulations.SC_Ymgap*100
N_PARAMS = 14

def estimate_electrical_cost(params,
                             yoke_type,
                             Ymgap = 0.,
                             z_gap = 10,
                            materials_directory=MATERIALS_DIR,
                            electricity_costs = 5.0,
                            NI_from_B = True):
    '''Estimate material quantities for the magnet 1 template without running simulation.
    
    :params parameters:
       The parameters of one magnet

    :return:
       A tuple containing (M_coil, Q, current_density)
    '''
    mag_params = magnet_simulations.get_magnet_params(params,Ymgap=Ymgap,yoke_type=yoke_type, B_goal = 1.9 if NI_from_B else params[13], materials_directory=materials_directory, z_gap=z_gap)
    coil_material = mag_params['coil_material']
    with open(join(materials_directory, coil_material)) as f:
        conductor_material_data = json.load(f)
    
    coil_radius = 0.5*mag_params['coil_diam(mm)']*1e-3
    ins = mag_params['insulation(mm)']*1e-3
    J_tar = mag_params['J_tar(A/mm2)']*1e6
    if J_tar<0:
        kappa_cu = conductor_material_data["material_cost(CHF/kg)"]
        kappa_cu += conductor_material_data["manufacturing_cost(CHF/kg)"]
        dens = conductor_material_data["density(g/m3)"]
        rho = conductor_material_data["resistivity(Ohm.m)"]
        J_tar = np.sqrt(kappa_cu*dens*1e-3/electricity_costs/rho)
        
    max_turns = int(mag_params['max_turns'])
    yoke_spacer = mag_params['yoke_spacer(mm)']*1e-3

    current = mag_params['NI(A)']

    # Extract parameters from mag_params instead of using raw params array
    X_mgap_1 = mag_params['Xmgap1(m)']
    X_mgap_2 = mag_params['Xmgap2(m)']
    X_core_1 = mag_params['Xcore1(m)']
    X_void_1 = mag_params['Xvoid1(m)']
    X_yoke_1 = mag_params['Xyoke1(m)']
    X_core_2 = mag_params['Xcore2(m)']
    X_void_2 = mag_params['Xvoid2(m)']
    X_yoke_2 = mag_params['Xyoke2(m)']
    Y_core_1 = mag_params['Ycore1(m)']
    Y_void_1 = mag_params['Yvoid1(m)']
    Y_yoke_1 = mag_params['Yyoke1(m)']
    Y_core_2 = mag_params['Ycore2(m)']
    Y_void_2 = mag_params['Yvoid2(m)']
    Y_yoke_2 = mag_params['Yyoke2(m)']
    Z_len = mag_params['Z_len(m)']


    Z_pos = 0
    


    # Make coil objects to calculate parameters
    # Determine the slot size
    slot_size = 2*min(Y_core_1, Y_core_2)


    # Determine number of conductors
    num_cond = np.int32(slot_size/2/(coil_radius+ins))
    if num_cond > max_turns:
        num_cond = max_turns
    assert num_cond > 0, f"No conductors fit in slot! slot_size={slot_size}, coil_radius={coil_radius}, ins={ins}, max_turns={max_turns}, Y_core_1={Y_core_1}, Y_core_2={Y_core_2}"
    # Vertical positions
    y = np.linspace(-0.5*slot_size + coil_radius + ins,
                   0.5*slot_size - coil_radius - ins, num_cond)

    # Calculate turn perimeter
    if yoke_type == "Mag1":
        if X_mgap_1 == 0.0 or X_mgap_2 == 0.0:
            # Single coil around core
            kp = np.array([[-X_core_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len             ],
                       [-X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                       [ X_core_2,          Z_pos + Z_len + yoke_spacer + ins + coil_radius    ],
                       [ X_core_2 + yoke_spacer + ins + coil_radius,   Z_pos + Z_len           ],
                       [ X_core_1 + yoke_spacer + ins + coil_radius,   Z_pos                   ],
                       [ X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                       [-X_core_1,                       Z_pos-yoke_spacer - ins - coil_radius ],
                       [-X_core_1 - yoke_spacer - ins - coil_radius,   Z_pos                   ]])
            
            coil = RacetrackCoil(kp, y, coil_radius, current/num_cond)
            turn_perimeter = coil.get_length()
        else:
            # Two coils around core
            kp_1 = np.array([[X_mgap_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len + coil_radius],
                           [X_mgap_2, Z_pos + Z_len + yoke_spacer + ins + coil_radius],
                           [X_core_2, Z_pos + Z_len + yoke_spacer + ins + coil_radius],
                           [X_core_2 + yoke_spacer + ins + coil_radius, Z_pos + Z_len],
                           [X_core_1 + yoke_spacer + ins + coil_radius, Z_pos],
                           [X_core_1, Z_pos-yoke_spacer - ins - coil_radius],
                           [X_mgap_1, Z_pos-yoke_spacer - ins - coil_radius],
                           [X_mgap_1 - yoke_spacer - ins - coil_radius, Z_pos]])
            
            kp_2 = kp_1.copy()
            kp_2[:, 0] *= -1.0
            
            coil1 = RacetrackCoil(kp_1, y, coil_radius, current/num_cond)
            coil2 = RacetrackCoil(kp_2, y, coil_radius, current/num_cond)
            turn_perimeter = coil1.get_length() + coil2.get_length()
    
    elif yoke_type == "Mag2":
        # Single coil around core for Mag2
        kp = np.array([[-X_core_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len],
                      [-X_core_2, Z_pos + Z_len + yoke_spacer + ins + coil_radius],
                      [X_core_2, Z_pos + Z_len + yoke_spacer + ins + coil_radius],
                      [X_core_2 + yoke_spacer + ins + coil_radius, Z_pos + Z_len],
                      [X_core_1 + yoke_spacer + ins + coil_radius, Z_pos],
                      [X_core_1, Z_pos-yoke_spacer - ins - coil_radius],
                      [-X_core_1, Z_pos-yoke_spacer - ins - coil_radius],
                      [-X_core_1 - yoke_spacer - ins - coil_radius, Z_pos]])
        
        coil = RacetrackCoil(kp, y, coil_radius, current/num_cond)
        turn_perimeter = coil.get_length()
    
    elif yoke_type == "Mag3":
        # Two coils on legs for Mag3
        kp_1 = np.array([[X_void_2 - yoke_spacer - ins - coil_radius, Z_pos + Z_len],
                       [X_void_2, Z_pos + Z_len + yoke_spacer + ins + coil_radius],
                       [X_yoke_2, Z_pos + Z_len + yoke_spacer + ins + coil_radius],
                       [X_yoke_2 + yoke_spacer + ins + coil_radius, Z_pos + Z_len],
                       [X_yoke_1 + yoke_spacer + ins + coil_radius, Z_pos],
                       [X_yoke_1, Z_pos-yoke_spacer - ins - coil_radius],
                       [X_void_1, Z_pos-yoke_spacer - ins - coil_radius],
                       [X_void_1 - yoke_spacer - ins - coil_radius, Z_pos]])
        
        kp_2 = kp_1.copy()
        kp_2[:, 0] *= -1.0
        
        coil1 = RacetrackCoil(kp_1, y, coil_radius, current/num_cond)
        coil2 = RacetrackCoil(kp_2, y, coil_radius, current/num_cond)
        turn_perimeter = coil1.get_length() + coil2.get_length()
    # Calculate horizontal slot size
    slot_size_horz = min(X_void_1 - X_core_1, X_void_2 - X_core_2)
    
    # Available space for coils
    A_geo = slot_size_horz*slot_size

    A_cu = abs(current)/J_tar

    # Coil size from target current density
    #A_coil = abs(current)/J_tar/conductor_material_data["filling_factor"]
    
    

    # Current density (for monitoring)
    #current_density = current/min([A_geo, A_coil])/conductor_material_data["filling_factor"]

    # Coil mass
    M_coil = A_cu*turn_perimeter*conductor_material_data['density(g/m3)']

    # Power consumption
    Q = current*current*turn_perimeter*conductor_material_data['resistivity(Ohm.m)']/A_cu

    
    C_coil = 1e-3*M_coil*(conductor_material_data["material_cost(CHF/kg)"]
                     +  conductor_material_data["manufacturing_cost(CHF/kg)"])
    
    C_edf = Q*electricity_costs

    return C_coil + C_edf




def get_iron_cost(phi, Ymgap = 0,zGap = 10, material = 'aisi1010.json', materials_directory=MATERIALS_DIR):
    '''Get the weight of the muon shield.'''

    dZ = phi[0] - zGap/2
    dX = phi[1]
    dX2 = phi[2]
    dY = phi[3]
    dY2 = phi[4]
    gap = phi[5]
    gap2 = phi[6]
    ratio_yoke_1 = phi[7]
    ratio_yoke_2 = phi[8]
    dY_yoke_1 = phi[9]
    dY_yoke_2 = phi[10]
    X_mgap_1 = phi[11]
    X_mgap_2 = phi[12]


    with open(join(materials_directory,material)) as f:
        iron_material_data = json.load(f)
    density = iron_material_data['density(g/m3)']*1E-9

    from scipy.spatial import ConvexHull
    def compute_solid_volume(vertices):
        """Compute the volume of the convex solid formed by two non-aligned rectangles using ConvexHull."""
        hull = ConvexHull(vertices)
        return hull.volume


    
    volume = 0
    corners = np.array([
        [X_mgap_1+dX, 0, 0],
        [X_mgap_1 + dX, dY, 0],
        [0, dY, 0],
        [0, 0, 0],
        [X_mgap_2+dX2,0, 2*dZ],
        [X_mgap_2+dX2, dY2, 2*dZ],
        [0, dY2, 2*dZ],
        [0, 0, 2*dZ]
    ])
    volume += compute_solid_volume(corners)
    corners = np.array([
        [X_mgap_1 + dX + gap, 0, 0],
        [X_mgap_1 + dX + gap + dX * ratio_yoke_1, 0, 0],
        [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY + Ymgap, 0],
        [X_mgap_1 + dX + gap, dY + Ymgap, 0],
        [X_mgap_2 + dX2 + gap2, 0, 2 * dZ],
        [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, 0, 2 * dZ],
        [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2 + Ymgap, 2 * dZ],
        [X_mgap_2 + dX2 + gap2, dY2 + Ymgap, 2 * dZ],
    ])
    volume += compute_solid_volume(corners)

    corners = np.array([
        [X_mgap_1, dY, 0],
        [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY, 0],
        [X_mgap_1 + dX + gap + dX * ratio_yoke_1, dY + dY_yoke_1, 0],
        [X_mgap_1, dY + dY_yoke_1, 0],
        [X_mgap_2, dY2, 2 * dZ],
        [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2, 2 * dZ],
        [X_mgap_2 + dX2 + gap2 + dX2 * ratio_yoke_2, dY2 + dY_yoke_2, 2 * dZ],
        [X_mgap_2, dY2 + dY_yoke_2, 2 * dZ],
    ])
    volume += compute_solid_volume(corners)
    
    M_iron = 4*volume*density
    C_iron = M_iron*(iron_material_data["material_cost(CHF/kg)"]
                     +  iron_material_data["manufacturing_cost(CHF/kg)"])
    print('The iron mass is = {:.2f} kg'.format(M_iron))   
    return C_iron




def get_field(resimulate_fields = False,
            params = None,
            file_name = 'data/outputs/fields.pkl',
            only_grid_params = False,
            **kwargs_field):
    '''Returns the field map for the given parameters. If from_file is True, the field map is loaded from the file_name.'''
    if resimulate_fields:
        fields = magnet_simulations.simulate_field(params, file_name = file_name,**kwargs_field)['B']
        d_space = kwargs_field['d_space']
    elif exists(file_name):
        print('Using field map from file', file_name)
        with h5py.File(file_name, 'r') as f:
            fields = f["B"][:]
            d_space = f["d_space"][:].tolist()
        d_space = [[round(float(val), 2) for val in axis] for axis in d_space]
    
    if only_grid_params: 
        fields = {'B': file_name,
                'range_x': [0.,d_space[0][1], d_space[0][2]],
                'range_y': [0.,d_space[1][1], d_space[1][2]],
                'range_z': [d_space[2][0],d_space[2][1], d_space[2][2]]}
    return fields


def CreateArb8(arbName, medium, dZ, corners, magField, field_profile,
               tShield, x_translation, y_translation, z_translation, stepGeo):

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
def CreateTarget(z_start:float):
    target_components = []
    N = 0#13
    T = 18#5
    materials = N * ["G4_Mo"] + T * ["G4_W"]
    lengths = [8., 2.5, 2.5, 2.5, 2.5, 
        2.5, 2.5, 2.5, 5.0, 5.0, 
        6.5, 8., 8., 5., 8., 
        10., 20., 50.]#35.]
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
                  middleGap2,ratio_yoke_1, ratio_yoke_2, dY_yoke_1,dY_yoke_2, gap,
                  gap2, Z, stepGeo, Ymgap = 0):
    dY += Ymgap #by doing in this way, the gap is filled with iron in Geant4, but simplifies
    coil_gap = gap
    coil_gap2 = gap2
    anti_overlap = 0.0# gap between fields in the corners for mitred joints (Geant goes crazy when
    # they touch each other)


    cornersMainL = np.array([
        middleGap, 
        -(dY +dY_yoke_1)- anti_overlap, 
        middleGap, 
        dY + dY_yoke_1- anti_overlap,
        dX + middleGap, 
        dY- anti_overlap, 
        dX + middleGap,
        -(dY- anti_overlap),
        middleGap2,
        -(dY2 + dY_yoke_2- anti_overlap), middleGap2, 
        dY2 + dY_yoke_2- anti_overlap,
        dX2 + middleGap2, 
        dY2- anti_overlap, 
        dX2 + middleGap2,
        -(dY2- anti_overlap)])

    cornersTL = np.array((middleGap + dX,dY,
                            middleGap,
                            dY + dY_yoke_1,
                            dX + ratio_yoke_1*dX + middleGap + coil_gap,
                            dY + dY_yoke_1,
                            dX + middleGap + coil_gap,
                            dY,
                            middleGap2 + dX2,
                            dY2,
                            middleGap2,
                            dY2 + dY_yoke_2,
                            dX2 + ratio_yoke_2*dX2 + middleGap2 + coil_gap2,
                            dY2 + dY_yoke_2,
                            dX2 + middleGap2 + coil_gap2,
                            dY2))

    cornersMainSideL = np.array((dX + middleGap + gap,
                                 -(dY), 
                                 dX + middleGap + gap,
                                dY, 
                                dX + ratio_yoke_1*dX + middleGap + gap, 
                                dY + dY_yoke_1,
                                dX + ratio_yoke_1*dX + middleGap + gap, 
                                -(dY + dY_yoke_1), 
                                dX2 + middleGap2 + gap2,
                                -(dY2), 
                                dX2 + middleGap2 + gap2, 
                                dY2,
                                dX2 + ratio_yoke_2*dX2 + middleGap2 + gap2, 
                                dY2 + dY_yoke_2, 
                                dX2 + ratio_yoke_2*dX2 + middleGap2 + gap2,
                                -(dY2 + dY_yoke_2)))

    
    cornersMainR = np.zeros(16, np.float16)
    cornersCLBA = np.zeros(16, np.float16)
    cornersMainSideR = np.zeros(16, np.float16)
    cornersCLTA = np.zeros(16, np.float16)
    cornersCRBA = np.zeros(16, np.float16)
    cornersCRTA = np.zeros(16, np.float16)

    cornersTR = np.zeros(16, np.float16)
    cornersBL = np.zeros(16, np.float16)
    cornersBR = np.zeros(16, np.float16)


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


def construct_block(medium, tShield,field_profile, stepGeo, length = 54):
    #iron block before the magnet
    z_gap = 1.
    dX = 50.
    dY = 50.
    dZ = length/2 - z_gap/2
    Z = -length/2
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


def design_muon_shield(params,fSC_mag = True, simulate_fields = False, field_map_file = None, cores_field:int = 1,extra_magnet = False, NI_from_B = True, use_diluted = False, SND = False):

    
    n_magnets = 7 + int(extra_magnet)
    cm = 1
    m = 100 * cm
    tesla = 1
    zgap = Z_GAP*cm

    magnetName = ["MagnAbsorb2", "Magn1", "Magn2", "Magn3", "Magn4", "Magn5", "Magn6", "Magn7"]

    fieldDirection = ["up", "up", "up", "up", "down", "down", "down", "down"]


    dZ1 = params[0]
    dZ2 = params[1]
    dZ3 = params[2]
    dZ4 = params[3]
    dZ5 = params[4]
    dZ6 = params[5]
    dZ7 = params[6]
    fMuonShieldLength = 2 * (dZ1 + dZ2 + dZ3 + dZ4 + dZ5 + dZ6 + dZ7)+ zgap


    dXIn = np.zeros(n_magnets)
    dXOut = np.zeros(n_magnets)
    gapIn = np.zeros(n_magnets)
    dYIn = np.zeros(n_magnets)
    dYOut = np.zeros(n_magnets)
    gapOut = np.zeros(n_magnets)
    dZf = np.zeros(n_magnets)
    ratio_yokesIn = np.ones(n_magnets)
    ratio_yokesOut = np.ones(n_magnets)
    dY_yokeIn = np.zeros(n_magnets)
    dY_yokeOut = np.zeros(n_magnets)

    Z = np.zeros(n_magnets)
    midGapIn= np.zeros(n_magnets)
    midGapOut= np.zeros(n_magnets)
    NI = np.zeros(n_magnets)


    offset = n_magnets - int(extra_magnet)
    n_params = len(params)/n_magnets - 1
    n_params = 13

    for i in range(n_magnets - int(extra_magnet)):
        dXIn[i] = params[offset + i * n_params]
        dXOut[i] = params[offset + i * n_params + 1]
        dYIn[i] = params[offset + i * n_params + 2]
        dYOut[i] = params[offset + i * n_params + 3]
        gapIn[i] = params[offset + i * n_params + 4]
        gapOut[i] = params[offset + i * n_params + 5]
        ratio_yokesIn[i] = params[offset + i * n_params + 6]
        ratio_yokesOut[i] = params[offset + i * n_params + 7]
        dY_yokeIn[i] = params[offset + i * n_params + 8]
        dY_yokeOut[i] = params[offset + i * n_params + 9]
        midGapIn[i] = params[offset + i * n_params + 10]
        midGapOut[i] = params[offset + i * n_params + 11]
        NI[i] = params[offset + i * n_params + 12]
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
        ratio_yokesIn[7] = ratio_yokesOut[6]
        ratio_yokesOut[7] = ratio_yokesIn[7]
        dY_yokeIn[7] = dY_yokeOut[6]
        dY_yokeOut[7] = dY_yokeIn[7]
        midGapIn[7] = midGapOut[6]
        midGapOut[7] = midGapIn[7]
        NI[7] = NI[6]
        dZf[7] = 0.1 * m
        Z[7] = Z[6] + dZf[6] + dZf[7]
        fMuonShieldLength += dZf[7]

    tShield = {
        'dz': fMuonShieldLength/100,
        'magnets':[],
        'global_field_map': {'B': np.array([])},
    }
    if field_map_file is not None or simulate_fields: 
        simulate_fields = (not exists(field_map_file)) or simulate_fields
        max_x = max(np.max(dXIn + dXIn * ratio_yokesIn + gapIn+midGapIn), np.max(dXOut + dXOut * ratio_yokesOut+gapOut+midGapOut))/100
        max_y = max(np.max(dYIn + dY_yokeIn), np.max(dYOut + dY_yokeOut))/100
        max_x = np.round(max_x,decimals=1).item()
        max_y = np.round(max_y,decimals=1).item()
        d_space = ((0,max_x+0.3, RESOL_DEF[0]), (0,max_y+0.3, RESOL_DEF[1]), (-0.5, np.ceil((Z[-1]+dZf[-1]+50+10)/100).item(), RESOL_DEF[2]))
        field_map = get_field(simulate_fields,np.asarray(params),Z_init = (Z[0] - dZf[0]), fSC_mag=fSC_mag,  d_space = d_space,
                              file_name=field_map_file, only_grid_params=True, NI_from_B_goal = NI_from_B, z_gap=zgap/100,
                              cores = min(cores_field,n_magnets), use_diluted = use_diluted)
        tShield['global_field_map'] = field_map
        #tShield['cost'] = cost
    cost = 0
    for nM in range(0,n_magnets):
        if dZf[nM] < 1 or dXIn[nM] < 1: continue
        if fSC_mag and (nM in [1,3]):
            continue
        if nM == 2 and fSC_mag:
            Ymgap = SC_Ymgap*cm
            ironField_s = 5.1 * tesla
        elif nM == 0:
            Ymgap = 0
            ironField_s = 1.9 * tesla
        else:
            Ymgap = 0
            ironField_s = 1.9 * tesla

        if simulate_fields or field_map_file is not  None:
            field_profile = 'global'
            fields_s = [[],[],[]]
        else:
            field_profile = 'uniform'
            if fieldDirection[nM] == "down":
                ironField_s = -ironField_s
            if nM in [4,5,6]: ironField_s *= ratio_yokesIn[nM]
            magFieldIron_s = [0., ironField_s, 0.]
            RetField_s = [0., -ironField_s/ratio_yokesIn[nM], 0.]
            ConRField_s = [-ironField_s/ratio_yokesIn[nM], 0., 0.]
            ConLField_s = [ironField_s/ratio_yokesIn[nM], 0., 0.]
            fields_s = [magFieldIron_s, RetField_s, ConRField_s, ConLField_s]

        create_magnet(magnetName[nM], "G4_Fe", tShield, fields_s, field_profile, dXIn[nM], dYIn[nM], dXOut[nM],
              dYOut[nM], dZf[nM], midGapIn[nM], midGapOut[nM], ratio_yokesIn[nM], ratio_yokesOut[nM],
              dY_yokeIn[nM], dY_yokeOut[nM], gapIn[nM], gapOut[nM], Z[nM], False, Ymgap=Ymgap)
        yoke_type = 'Mag1' if nM in [0,1,2,3] else 'Mag3'
        if fSC_mag and nM==2: yoke_type = 'Mag2'
        cost += get_iron_cost([dZf[nM]+zgap/2, dXIn[nM], dXOut[nM], dYIn[nM], dYOut[nM], gapIn[nM], gapOut[nM], ratio_yokesIn[nM], ratio_yokesOut[nM], dY_yokeIn[nM], dY_yokeOut[nM], midGapIn[nM], midGapOut[nM]], Ymgap=Ymgap, zGap=zgap)        
        cost += estimate_electrical_cost(np.array([dZf[nM]+zgap/2, dXIn[nM], dXOut[nM], dYIn[nM], dYOut[nM], gapIn[nM], gapOut[nM], ratio_yokesIn[nM], ratio_yokesOut[nM], dY_yokeIn[nM], dY_yokeOut[nM], midGapIn[nM], midGapOut[nM], NI[nM]]), Ymgap=Ymgap, z_gap=zgap, yoke_type=yoke_type, NI_from_B=NI_from_B)
    if SND and (midGapIn[5] > 30) and (midGapOut[5] > 30):
        gap = 10 * cm
        dX = 20.-gap
        dY = 20.-gap
        dZ = 172/2
        Z = (Z[5]+dZf[5])-dZ
        corners = np.array([
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
        CreateArb8('SND_Emu_Si', 'G4_Fe', dZ, corners, [0.,0.,0.], 'uniform', Block, 0, 0, Z, False)
        tShield['magnets'].append(Block)
    tShield['cost'] = cost
    return tShield





def get_design_from_params(params, 
                           fSC_mag:bool = True, 
                           force_remove_magnetic_field = False,
                           simulate_fields = False,
                           field_map_file = None,
                           sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6,'position':82},
                           add_cavern:bool = True,
                           add_target:bool = True,
                           cores_field:int = 1,
                           extra_magnet = False,
                           NI_from_B = True, 
                           use_diluted = False,
                           SND = False):
    params = np.round(params, 2)
    shield = design_muon_shield(params, fSC_mag, simulate_fields = simulate_fields, field_map_file = field_map_file, cores_field=cores_field, extra_magnet = extra_magnet, NI_from_B = NI_from_B, use_diluted=use_diluted, SND=SND)
    shift = -2.14#-2.345
    cavern_transition = 20.518+shift #m
    World_dZ = 200 #m
    World_dX = World_dY = 30 if add_cavern else 15
    
    construct_block("G4_Fe", shield, 'global' if simulate_fields else 'uniform', False)
    if add_cavern: shield["cavern"] = CreateCavern(cavern_transition, length = World_dZ)
    if add_target: shield['target'] = CreateTarget(z_start=shift)
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
    if seed is None: seeds = (np.random.randint(256), np.random.randint(256), np.random.randint(256), np.random.randint(256))
    else: seeds = (seed, seed, seed, seed)
    detector = json.dumps(detector,default=lambda o: float(o) if isinstance(o, np.float32) else o)
    output_data = initialize(*seeds,detector) #
    return output_data

if __name__ == '__main__':
    import json
    import numpy as np
    import os
    file_map_file = 'data/outputs/fields_test'
    t1 = time()
    from lib.reference_designs.params import sc_v6, optimal_oliver
    params = [120.50, 127.5, 250., 372.5, 203.76, 200.59, 198.83, 
              50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00, 1.0, 50.00,  50.00, 0., 0.0, 0.,
              72.00,  51.00,  29.00,  46.00,  10.00,   7.00, 1.00, 1.0, 72.00,  51.00, 0.0, 0.00, 0.,
              54.00,  38.00,  46.00, 130.00,  14.00,   9.00, 1.00, 1.0, 54.00,  38.00, 0.0, 0.00, 0.,
              10.00,  31.00,  35.00,  31.00,  51.00,  11.00, 1.00, 1.0, 10.00,  31.00, 0.0, 0.00, 0.,
               5.00,  32.00,  54.00,  24.00,   8.00,   8.00, 1.00, 1.0,  5.00,  32.00, 0.0, 0.00, 0.,
              22.00,  32.00, 130.00,  35.00,   8.00,  13.00, 1.00, 1.0, 22.00,  32.00, 0.2, 0.0, 0.,
              33.00,  77.00,  85.00,  90.00,   9.00,  26.00, 1.00, 1.0, 33.00,  77.00, 0.0, 0.00, 0.]
    fSC_mag = False
    core_field = 8
    use_diluted =  False
    detector = get_design_from_params(params, simulate_fields=True,field_map_file = file_map_file, add_cavern=True, cores_field=core_field, fSC_mag = fSC_mag, use_diluted = use_diluted)
    t1_init = time()
    output_data = initialize_geant4(detector)
    print('Time to initialize', time()-t1_init)
    print('TOTAL TIME', time()-t1)
