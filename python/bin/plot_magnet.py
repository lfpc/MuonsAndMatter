import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lib.ship_muon_shield_customfield import get_design_from_params, filter_fields
from lib.magnet_simulations import get_symmetry

def plot_magnet(detector, 
                output_file='plots/detector_visualization.png',
                muon_data = [], 
                sensitive_film_position = None,
                fixed_zlim:bool = False, 
                azim:float = 126,
                elev:float = 17,
                ignore_magnets = []):
    magnets = detector['magnets']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if ('B' in detector['global_field_map']) and np.size(detector['global_field_map']['B']) > 0:
        points = np.meshgrid(np.arange(*detector['global_field_map']['range_x']),
                          np.arange(*detector['global_field_map']['range_y']),
                            np.arange(*detector['global_field_map']['range_z']))
        points = np.column_stack((points[0].ravel(), points[1].ravel(), points[2].ravel()))
        detector['global_field_map'] = get_symmetry(points,np.asarray(detector['global_field_map']['B']), reorder=True)
    if "sensitive_film" in detector  and sensitive_film_position is not None:
        cz, cx, cy = detector["sensitive_film"]["z_center"], 0, 0
        if sensitive_film_position is not None: 
            if isinstance(sensitive_film_position,tuple):
                cz = sensitive_film_position[1]+detector['magnets'][sensitive_film_position[0]]['z_center']+detector['magnets'][sensitive_film_position[0]]['dz']
            else: cz = sensitive_film_position+detector['magnets'][-1]['z_center']+detector['magnets'][-1]['dz']

        # Calculate the half-sizes
        hw = detector["sensitive_film"]["dx"] / 2
        hl = detector["sensitive_film"]["dy"] / 2
        hh = detector["sensitive_film"]["dz"] / 2

        # Define the vertices of the box
        vertices = np.array([
            [cz - hh, cx - hw, cy - hl, ],
            [cz - hh, cx + hw, cy - hl, ],
            [cz - hh, cx + hw, cy + hl, ],
            [cz - hh, cx - hw, cy + hl, ],
            [cz + hh, cx - hw, cy - hl, ],
            [cz + hh, cx + hw, cy - hl, ],
            [cz + hh, cx + hw, cy + hl, ],
            [cz + hh, cx - hw, cy + hl, ],
        ])

        # Define the edges of the box
        edges = [
            [vertices[j] for j in [0, 1, 2, 3]],  # bottom face
            [vertices[j] for j in [4, 5, 6, 7]],  # top face
            [vertices[j] for j in [0, 1, 5, 4]],  # front face
            [vertices[j] for j in [2, 3, 7, 6]],  # back face
            [vertices[j] for j in [1, 2, 6, 5]],  # right face
            [vertices[j] for j in [0, 3, 7, 4]],  # left face
        ]
        box = Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='orange', alpha=.15)
        ax.add_collection3d(box)
    for i,mag in enumerate(magnets):
        if i in ignore_magnets: continue
        #print(f'MAG {i}' , mag)
        z1 = mag['z_center']-mag['dz']
        z2 = mag['z_center']+mag['dz']

        for i, component in enumerate(mag['components']):
            the_dat = component['corners']
            if component['field_profile'] == 'uniform':field = component['field']
            elif component['field_profile'] == 'global': 
                field = filter_fields(*detector['global_field_map'],component['corners'],component['z_center'],component['dz'])
                field = np.mean(field[1],axis=0)
            else: field = np.mean(component['field'][1],axis=0)
            B_th = 0.7
            if field[1] < -B_th:
                col = 'red'
            elif field[1] > B_th:
                col = 'green'
            elif field[0] < -B_th:
                col = 'red'
            elif field[0] > B_th:
                col = 'green'
            elif field[2] < -B_th:
                col = 'blue'
            elif field[2] > B_th:
                col = 'blue'
            else: col = 'grey'

            corners = np.array(
                [
                    [the_dat[0], the_dat[1], z1], [the_dat[2], the_dat[3], z1], [the_dat[4], the_dat[5], z1], [the_dat[6], the_dat[7], z1],
                    [the_dat[0 + 8], the_dat[1 + 8], z2], [the_dat[2 + 8], the_dat[3 + 8], z2], [the_dat[4 + 8], the_dat[5 + 8], z2], [the_dat[6 + 8], the_dat[7 + 8], z2],
                    ]
            )
            
            corners = np.array([[c[2], c[0], c[1]] for c in corners])
            # Define the 12 edges connecting the corners
            edges = [[corners[j] for j in [0, 1, 2, 3]],
                        [corners[j] for j in [4, 5, 6, 7]],
                        [corners[j] for j in [0, 1, 5, 4]],
                        [corners[j] for j in [2, 3, 7, 6]],
                        [corners[j] for j in [0, 3, 7, 4]],
                        [corners[j] for j in [1, 2, 6, 5]]]

            # # Plot the edges
            ax.add_collection3d(Poly3DCollection(edges, facecolors=col, linewidths=0.3, edgecolors='r', alpha=0.2))
            #
            # # Scatter plot of the corners
            # ax.scatter3D(corners[:, 0], corners[:, 1], corners[:, 2], color='b', s=0.04)

    colors = plt.cm.get_cmap('tab10', 10)  # Get a colormap with 10 colors
    #total_sensitive_hits = 0
    for i, data in enumerate(muon_data):
        if isinstance(data,dict):
            x = data['x']
            y = data['y']
            z = data['z']

            if 'pdg_id' in data: 
                particle = data['pdg_id']  
            else: particle = 13
        else:
            _,_,_,x,y,z,particle = data
            if sensitive_film_position is not None: z = sensitive_film_position*np.ones_like(z)+detector['magnets'][-1]['z_center']+detector['magnets'][-1]['dz']
        
        #total_sensitive_hits += 1
        ax.scatter(z[particle>0], x[particle>0], y[particle>0], color='blue', label=f'Muon {i + 1}', s=0.5)
        ax.scatter(z[particle<0], x[particle<0], y[particle<0], color='orange', label=f'AntiMuon {i + 1}', s=0.5)
    
    # Plot cavern
    if "cavern" in detector:
        for cavern in detector["cavern"]:
            for component in cavern["components"]:
                the_dat = np.array(component)#.reshape(-1, 2)
                z1 = max(-15,cavern['z_center'] - cavern['dz'])
                z2 = min(30,cavern['z_center'] + cavern['dz'])
                corners = np.array(
                [
                    [the_dat[0], the_dat[1], z1], [the_dat[2], the_dat[3], z1], [the_dat[4], the_dat[5], z1], [the_dat[6], the_dat[7], z1],
                    [the_dat[0 + 8], the_dat[1 + 8], z2], [the_dat[2 + 8], the_dat[3 + 8], z2], [the_dat[4 + 8], the_dat[5 + 8], z2], [the_dat[6 + 8], the_dat[7 + 8], z2],
                    ]
            )
            
                corners = np.array([[c[2], c[0], c[1]] for c in corners])
                                # Define the 12 edges connecting the corners
                edges = [[corners[j] for j in [0, 1, 2, 3]],
                            [corners[j] for j in [4, 5, 6, 7]],
                            [corners[j] for j in [0, 1, 5, 4]],
                            [corners[j] for j in [2, 3, 7, 6]],
                            [corners[j] for j in [0, 3, 7, 4]],
                            [corners[j] for j in [1, 2, 6, 5]]]
                ax.add_collection3d(Poly3DCollection(edges, facecolors='gray', linewidths=0.07, edgecolors='black', alpha=0.25))
    # Plot horizontal plane at y = -1.7
    z_plane = -1.7
    x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    y_range = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_plane)
    ax.plot_surface(X, Y,Z, color='black', alpha=0.3)

    if fixed_zlim: ax.set_xlim(-30+sensitive_film_position, detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz']-5)
    else: ax.set_xlim(40, -5)
    ax.set_ylim(-5, 10)
    ax.set_zlim(-5, 10)

    # Adjust the view angle and zoom level
    # ax.view_init(elev=20., azim=30)  # Adjust elevation and azimuth
    # ax.dist = 6 # Smaller values zoom in, larger values zoom out

    ax.set_xlabel('Z (m)')
    ax.set_ylabel('X (m)')
    ax.set_zlabel('Y (m)')
    #ax.view_init(elev=17., azim=126)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    if output_file is not None and output_file != '':
        fig.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
        print(f'Plot saved to {output_file}')
    #print("Total sensitive hits plotted", total_sensitive_hits)
    plt.close()

def construct_and_plot(muons, 
        phi, 
        fSC_mag:bool = True,
        sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6,'position':83.2},
        use_field_maps = False,
        field_map_file = None,
        cavern = True,
        kwargs_plot = {}):
    detector = get_design_from_params(params = phi,fSC_mag = fSC_mag, use_field_maps=use_field_maps, field_map_file = field_map_file, sensitive_film_params=sensitive_film_params, add_cavern=cavern)
    plot_magnet(detector,
                muon_data = muons, 
                sensitive_film_position = sensitive_film_params['position'],#sensitive_film_params['position'], 
                **kwargs_plot)
    