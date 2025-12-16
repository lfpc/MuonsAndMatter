import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lib.ship_muon_shield_customfield import get_design_from_params
from lib.magnet_simulations import get_symmetry
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_fields(points, fields, output_file='plots/fields.png'):
    # Filter points where y is approximately 0 for left plot, x is approximately 0 for right plot
    tolerance = 1e-6
    mask_y0 = np.abs(points[:, 1]) < tolerance
    mask_x0 = np.abs(points[:, 0]) < tolerance

    points_xz = points[mask_y0]
    fields_xz = fields[mask_y0]
    points_yz = points[mask_x0]
    fields_yz = fields[mask_x0]

    # Use y-component of the field
    fields_xz = fields_xz[:, 1]
    fields_yz = fields_yz[:, 1]

    # Custom colormap: green for negative, white at zero, red for positive
    colors = [(0, 0.8, 0), (1, 1, 1), (0.8, 0, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('green_white_red', colors)
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=3)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    # Left: X vs Z (Y=0)
    tcf1 = axes[0].tricontourf(points_xz[:, 0], points_xz[:, 2], fields_xz, cmap=custom_cmap, levels=70, norm=norm)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Z')
    axes[0].set_title('Field Y-Component for Y=0')
    axes[0].grid(True)
    # Right: Y vs Z (X=0)
    tcf2 = axes[1].tricontourf(points_yz[:, 1], points_yz[:, 2], fields_yz, cmap=custom_cmap, levels=70, norm=norm)
    axes[1].set_xlabel('Y')
    axes[1].set_title('Field Y-Component for X=0')
    axes[1].grid(True)

    # Shared colorbar
    cbar = fig.colorbar(tcf1, ax=axes, orientation='vertical', fraction=0.025, pad=0.04)
    cbar.set_label('Field Y-Component')

    #plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
    print(f'Plot saved to {output_file}')
    plt.close()


def plot_magnet(detector, 
                output_file='plots/detector_visualization.png',
                muon_data = [], 
                sensitive_film_position = None,
                azim:float = 126,
                elev:float = 17,
                plot_axis = True,
                ignore_magnets = []):
    magnets = detector['magnets']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not plot_axis:
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if ('B' in detector['global_field_map']) and np.size(detector['global_field_map']['B']) > 0:
        
        points = np.meshgrid(np.arange(*detector['global_field_map']['range_x']),
                          np.arange(*detector['global_field_map']['range_y']),
                            np.arange(*detector['global_field_map']['range_z']))
        points = np.column_stack((points[0].ravel(), points[1].ravel(), points[2].ravel()))
        #detector['global_field_map'] = get_symmetry(points,np.asarray(detector['global_field_map']['B']), reorder=True)

    if "sensitive_film" in detector and isinstance(detector["sensitive_film"], list):
        for sf_idx, sf in enumerate(detector["sensitive_film"]):
            cz, cx, cy = sf["z_center"], 0, 0
            if sensitive_film_position is not None:
                if isinstance(sensitive_film_position, (list, tuple)):
                    # If sensitive_film_position is a list/tuple, try to match index
                    if isinstance(sensitive_film_position, list) and len(sensitive_film_position) > sf_idx:
                        pos = sensitive_film_position[sf_idx]
                        if isinstance(pos, tuple):
                            cz = pos[1] + detector['magnets'][pos[0]]['z_center'] + detector['magnets'][pos[0]]['dz']
                        else:
                            cz = pos + detector['magnets'][-1]['z_center'] + detector['magnets'][-1]['dz']
                    elif isinstance(sensitive_film_position, tuple):
                        cz = sensitive_film_position[1] + detector['magnets'][sensitive_film_position[0]]['z_center'] + detector['magnets'][sensitive_film_position[0]]['dz']
                else:
                    cz = sensitive_film_position + detector['magnets'][-1]['z_center'] + detector['magnets'][-1]['dz']

            # Calculate the half-sizes
            hw = sf["dx"] / 2
            hl = sf["dy"] / 2
            hh = sf["dz"] / 2

            # Define the vertices of the box
            vertices = np.array([
                [cz - hh, cx - hw, cy - hl],
                [cz - hh, cx + hw, cy - hl],
                [cz - hh, cx + hw, cy + hl],
                [cz - hh, cx - hw, cy + hl],
                [cz + hh, cx - hw, cy - hl],
                [cz + hh, cx + hw, cy - hl],
                [cz + hh, cx + hw, cy + hl],
                [cz + hh, cx - hw, cy + hl],
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
    if "target" in detector:
        for target in detector["target"]['components']:
            z_center = target["z_center"]
            dz = target["dz"]
            radius = target["radius"]
            material = target["material"]

            # Define the color based on the material
            if material == "G4_W":
                color = 'darkblue'
            elif material == "G4_WATER":
                color = 'lightblue'
            elif material == "G4_Mo":
                color = 'brown'
            else:
                color = 'gray'  # Default color

            # Define the vertices of the cylinder
            z1 = z_center - dz/2
            z2 = z_center + dz/2
            theta = np.linspace(0, 2 * np.pi, 100)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)

            # Create the top and bottom circles
            top_circle = np.array([z2 * np.ones_like(theta), x, y]).T
            bottom_circle = np.array([z1 * np.ones_like(theta), x, y]).T

            # Create the side faces
            side_faces = []
            for i in range(len(theta) - 1):
                side_faces.append([bottom_circle[i], bottom_circle[i + 1], top_circle[i + 1], top_circle[i]])

            ax.add_collection3d(Poly3DCollection([top_circle], facecolors=color, linewidths=0.01, edgecolors=color, alpha=0.7))
            ax.add_collection3d(Poly3DCollection([bottom_circle], facecolors=color, linewidths=0.01, edgecolors=color, alpha=0.7))
            ax.add_collection3d(Poly3DCollection(side_faces, facecolors=color, linewidths=0.01, edgecolors = color, alpha=0.7))
    
    for m,mag in enumerate(magnets):
        if m in ignore_magnets: continue
        #print(f'MAG {i}' , mag)
        z1 = mag['z_center']-mag['dz']
        z2 = mag['z_center']+mag['dz']

        for i, component in enumerate(mag['components']):
            the_dat = component['corners']
            if component['field_profile'] == 'uniform': field = component['field']
            elif component['field_profile'] == 'global':
                #field = filter_fields(*detector['global_field_map'],component['corners'],component['z_center'],component['dz'])
                #field = np.mean(field[1],axis=0)
                field = [0.,0.,0.]
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

    #total_sensitive_hits = 0
    cmap_muon = cm.get_cmap('Blues')
    cmap_antimuon = cm.get_cmap('Oranges')
    norm = mcolors.Normalize(vmin=0, vmax=250)  
    for i, data in enumerate(muon_data):
        if isinstance(data,dict):
            x = np.array(data['x'])
            y = np.array(data['y'])
            z = np.array(data['z'])
            p = np.sqrt(np.array(data['px'])**2 + np.array(data['py'])**2 + np.array(data['pz'])**2)[0]
            particle = np.array(data['pdg_id'])
            alpha = 0.3

            
        else:
            px,py,pz,x,y,z,particle = data[:7]
            p = np.sqrt(px**2 + py**2 + pz**2)
            if sensitive_film_position is not None: z = sensitive_film_position*np.ones_like(z)+detector['magnets'][-1]['z_center']+detector['magnets'][-1]['dz']
            alpha = 0.3
        if np.all(particle>0):
            color = cmap_muon(norm(p))
        else:
            color = cmap_antimuon(norm(p))

        
        #total_sensitive_hits += 1
        ax.scatter(z[particle>0], x[particle>0], y[particle>0], color=color, label=f'Muon {i + 1}', s=0.1, alpha=alpha)
        ax.scatter(z[particle<0], x[particle<0], y[particle<0], color=color, label=f'AntiMuon {i + 1}', s=0.1, alpha=alpha)
    # Add colorbars
    if plot_axis:
        sm_muon = cm.ScalarMappable(cmap=cmap_muon, norm=norm)
        sm_antimuon = cm.ScalarMappable(cmap=cmap_antimuon, norm=norm)
        sm_muon.set_array([])
        cbar_muon = plt.colorbar(sm_muon, ax=ax, shrink=0.5)  # Reduced shrink and pad
        cbar_muon.set_label("Muons")
        sm_antimuon.set_array([])
        cbar_antimuon = plt.colorbar(sm_antimuon, ax=ax, shrink=0.5)  # Adjusted shrink and pad
        cbar_antimuon.set_label("Anti-muons")
        cbar_muon.ax.set_position([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar_antimuon.ax.set_position([0.88, 0.15, 0.02, 0.7])  # Move closer
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

        # Plot sensitive_box (Decay Vessel) if present
    if "sensitive_box" in detector:
        box = detector["sensitive_box"]
        # corners: flat list of 16 values (8 x/y pairs)
        corners = np.array(box["corners"]).reshape(8, 2)
        dz = box["dz"]
        z_center = box["z_center"]
        # Build 8 3D corners (first 4 at z_center-dz, next 4 at z_center+dz)
        z1 = z_center - dz
        z2 = z_center + dz
        box_corners = np.array([
            [z1, corners[0][0], corners[0][1]],
            [z1, corners[1][0], corners[1][1]],
            [z1, corners[2][0], corners[2][1]],
            [z1, corners[3][0], corners[3][1]],
            [z2, corners[4][0], corners[4][1]],
            [z2, corners[5][0], corners[5][1]],
            [z2, corners[6][0], corners[6][1]],
            [z2, corners[7][0], corners[7][1]],
        ])
        # Reorder to (z, x, y) for plotting
        box_corners = np.array([[c[0], c[1], c[2]] for c in box_corners])
        # Define the 12 unique edges of the box
        edge_indices = [
            (0,1), (1,2), (2,3), (3,0),  # bottom
            (4,5), (5,6), (6,7), (7,4),  # top
            (0,4), (1,5), (2,6), (3,7)   # verticals
        ]
        for i1, i2 in edge_indices:
            ax.plot(
                [box_corners[i1][0], box_corners[i2][0]],
                [box_corners[i1][1], box_corners[i2][1]],
                [box_corners[i1][2], box_corners[i2][2]],
                color='gray', linewidth=1, alpha=0.7
            )
    # Plot horizontal plane at y = -1.7
    if 80 < azim < 100 and -10 < elev < 20:
        z_plane1 = -1.7
        z_plane2 = -3.36
        x_split = 20.518 - 2.345

        # First plane
        x_range1 = np.linspace(ax.get_xlim()[0], x_split, 100)
        y_range1 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
        X1, Y1 = np.meshgrid(x_range1, y_range1)
        Z1 = np.full_like(X1, z_plane1)
        ax.plot_surface(X1, Y1, Z1, color='black', alpha=0.3)

        # Second plane
        x_range2 = np.linspace(x_split, ax.get_xlim()[1], 100)
        y_range2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
        X2, Y2 = np.meshgrid(x_range2, y_range2)
        Z2 = np.full_like(X2, z_plane2)
        ax.plot_surface(X2, Y2, Z2, color='black', alpha=0.3)

        z_plane1 = 5.8
        z_plane2 = 10.6
        Z1 = np.full_like(X1, z_plane1)
        ax.plot_surface(X1, Y1, Z1, color='black', alpha=0.3)
        X2, Y2 = np.meshgrid(x_range2, y_range2)
        Z2 = np.full_like(X2, z_plane2)
        ax.plot_surface(X2, Y2, Z2, color='black', alpha=0.3)
    elif azim < 10 or 80< elev<100 :
        y_plane1 = -3.57
        y_plane2 = -4.57
        x_split = 20.518 - 2.345

        # First plane
        x_range1 = np.linspace(ax.get_xlim()[0], x_split, 100)
        z_range1 = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 100)
        X1, Z1 = np.meshgrid(x_range1, z_range1)
        Y1 = np.full_like(X1, y_plane1)
        ax.plot_surface(X1, Y1, Z1, color='black', alpha=0.3)

        # Second plane
        x_range2 = np.linspace(x_split, ax.get_xlim()[1], 100)
        z_range2 = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 100)
        X2, Z2 = np.meshgrid(x_range2, z_range2)
        Y2 = np.full_like(X2, y_plane2)
        ax.plot_surface(X2, Y2, Z2, color='black', alpha=0.3)

        y_plane1 = 6.43
        y_plane2 = 11.43
        X1, Z1 = np.meshgrid(x_range1, z_range1)
        Y1 = np.full_like(X1, y_plane1)
        ax.plot_surface(X1, Y1, Z1, color='black', alpha=0.3)
        X2, Z2 = np.meshgrid(x_range2, z_range2)
        Y2 = np.full_like(X2, y_plane2)
        ax.plot_surface(X2, Y2, Z2, color='black', alpha=0.3)

    if sensitive_film_position is None and not 'sensitive_box' in detector: 
        ax.set_xlim(40, -5)
    elif sensitive_film_position[0] < 0: sensitive_film_position = None
    elif 'sensitive_box' in detector:
        ax.set_xlim(80, -5)
    else:
        ax.set_xlim(3+max(sensitive_film_position), -5)
    ax.set_ylim(-5, 7)
    ax.set_zlim(-5, 7)

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
        simulate_fields = False,
        field_map_file = None,
        decay_vessel:bool = False,
        cavern = True,
        SND = False,
        **kwargs_plot):
    detector = get_design_from_params(params = phi,fSC_mag = fSC_mag, simulate_fields=simulate_fields, field_map_file = field_map_file, sensitive_film_params=sensitive_film_params, add_cavern=cavern, sensitive_decay_vessel=decay_vessel, SND = SND)
    plot_magnet(detector,
                muon_data = muons, 
                sensitive_film_position = [sens['position'] for sens in sensitive_film_params],#sensitive_film_params['position'], 
                **kwargs_plot)

def plot_magnet_fancy(detector, 
                      output_file='plots/detector_fancy.pdf',
                      muon_data=[], 
                      sensitive_film_position=None,
                      azim: float = 126,
                      elev: float = 17,
                      ignore_magnets=[]):
    """
    Plots the detector geometry in a high-quality style for papers.
    Automatically detects if muon_data contains full tracks or point-cloud data.
    """
    
    # Initialize Figure with high resolution
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # --- STYLE: Clean white background ---
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # ==========================================
    # 1. GEOMETRY PLOTTING (Films, Targets, Magnets)
    # ==========================================
    
    # --- Sensitive Films ---
    if "sensitive_film" in detector and isinstance(detector["sensitive_film"], list):
        for sf_idx, sf in enumerate(detector["sensitive_film"]):
            cz, cx, cy = sf["z_center"], 0, 0
            if sensitive_film_position is not None:
                # Logic to determine position based on input or defaults
                if isinstance(sensitive_film_position, (list, tuple)):
                    if isinstance(sensitive_film_position, list) and len(sensitive_film_position) > sf_idx:
                        pos = sensitive_film_position[sf_idx]
                        if isinstance(pos, tuple):
                            cz = pos[1] + detector['magnets'][pos[0]]['z_center'] + detector['magnets'][pos[0]]['dz']
                        else:
                            cz = pos + detector['magnets'][-1]['z_center'] + detector['magnets'][-1]['dz']
                    elif isinstance(sensitive_film_position, tuple):
                        cz = sensitive_film_position[1] + detector['magnets'][sensitive_film_position[0]]['z_center'] + detector['magnets'][sensitive_film_position[0]]['dz']
                else:
                    cz = sensitive_film_position + detector['magnets'][-1]['z_center'] + detector['magnets'][-1]['dz']
            hw, hl, hh = sf["dx"] / 2, sf["dy"] / 2, sf["dz"] / 2
            
            # Create box vertices
            vertices = np.array([
                [cz - hh, cx - hw, cy - hl], [cz - hh, cx + hw, cy - hl],
                [cz - hh, cx + hw, cy + hl], [cz - hh, cx - hw, cy + hl],
                [cz + hh, cx - hw, cy - hl], [cz + hh, cx + hw, cy - hl],
                [cz + hh, cx + hw, cy + hl], [cz + hh, cx - hw, cy + hl],
            ])
            
            edges = [
                [vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]],
                [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]],
                [vertices[j] for j in [1, 2, 6, 5]], [vertices[j] for j in [0, 3, 7, 4]],
            ]
            box = Poly3DCollection(edges, facecolors='cyan', linewidths=0.5, edgecolors='teal', alpha=0.4)
            ax.add_collection3d(box)

    # --- Targets ---
    if "target" in detector:
        for target in detector["target"]['components']:
            z_center, dz, radius = target["z_center"], target["dz"], target["radius"]
            material = target["material"]
            
            color = {'G4_W': 'navy', 'G4_WATER': 'skyblue', 'G4_Mo': 'sienna'}.get(material, 'silver')

            z1, z2 = z_center - dz/2, z_center + dz/2
            theta = np.linspace(0, 2 * np.pi, 50)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)

            top_circle = np.array([z2 * np.ones_like(theta), x, y]).T
            bottom_circle = np.array([z1 * np.ones_like(theta), x, y]).T

            side_faces = []
            for i in range(len(theta) - 1):
                side_faces.append([bottom_circle[i], bottom_circle[i + 1], top_circle[i + 1], top_circle[i]])

            ax.add_collection3d(Poly3DCollection([top_circle], facecolors=color, linewidths=0, alpha=0.8, shade=True))
            ax.add_collection3d(Poly3DCollection([bottom_circle], facecolors=color, linewidths=0, alpha=0.8, shade=True))
            ax.add_collection3d(Poly3DCollection(side_faces, facecolors=color, linewidths=0.1, edgecolors=color, alpha=0.8, shade=True))

    # --- Magnets ---
    magnets = detector['magnets']
    for m, mag in enumerate(magnets):
        if m in ignore_magnets: continue
        z1 = mag['z_center'] - mag['dz']
        z2 = mag['z_center'] + mag['dz']

        for component in mag['components']:
            the_dat = component['corners']
            
            # Determine color based on field strength/direction
            if component['field_profile'] == 'uniform': field = component['field']
            elif component['field_profile'] == 'global': field = [0., 0., 0.]
            else: field = np.mean(component['field'][1], axis=0)
            
            B_th = 0.7
            if field[1] < -B_th: col = 'firebrick'
            elif field[1] > B_th: col = 'forestgreen'
            elif m<4: col = 'firebrick'
            else: col = 'forestgreen'

            corners_raw = np.array([
                [the_dat[0], the_dat[1], z1], [the_dat[2], the_dat[3], z1], 
                [the_dat[4], the_dat[5], z1], [the_dat[6], the_dat[7], z1],
                [the_dat[0+8], the_dat[1+8], z2], [the_dat[2+8], the_dat[3+8], z2], 
                [the_dat[4+8], the_dat[5+8], z2], [the_dat[6+8], the_dat[7+8], z2],
            ])
            
            # Permute to (Z, X, Y) for plotting
            corners = np.array([[c[2], c[0], c[1]] for c in corners_raw])
            
            edges = [
                [corners[j] for j in [0, 1, 2, 3]], [corners[j] for j in [4, 5, 6, 7]],
                [corners[j] for j in [0, 1, 5, 4]], [corners[j] for j in [2, 3, 7, 6]],
                [corners[j] for j in [0, 3, 7, 4]], [corners[j] for j in [1, 2, 6, 5]]
            ]

            ax.add_collection3d(Poly3DCollection(edges, facecolors=col, linewidths=0.5, edgecolors='black', alpha=0.3))
    
    if len(muon_data) > 0:
        cmap_muon = plt.get_cmap('Blues')
        cmap_antimuon = plt.get_cmap('Oranges')
        norm = mcolors.Normalize(vmin=0, vmax=250)
        
        # Determine mode based on first element
        is_track_mode = isinstance(muon_data[0], dict)

        if is_track_mode:
            # --- MODE A: TRACKS (Lines) ---
            for i, data in enumerate(muon_data):
                x = np.array(data['x'])
                y = np.array(data['y'])
                z = np.array(data['z'])
                p = np.sqrt(np.array(data['px'])**2 + np.array(data['py'])**2 + np.array(data['pz'])**2)[0]
                particle = np.array(data['pdg_id'])
                
                if np.all(particle > 0): 
                    color = cmap_muon(norm(p))
                else: 
                    color = cmap_antimuon(norm(p))

                # Plot lines/points
                ax.scatter(z, x, y, color=color, s=0.1, alpha=0.3)

        else:
            # --- MODE B: CLOUD (Spheres + Arrows) ---
            # We aggregate data first for vectorized plotting (much faster and cleaner)
            
            # Arrays to hold plotting data
            cloud_x, cloud_y, cloud_z = [], [], []
            cloud_u, cloud_v, cloud_w = [], [], [] # Vector directions
            cloud_c, cloud_s = [], [] # Colors and Sizes
            
            for data in muon_data:
                px, py, pz, x, y, z, particle = data[:7]
            
                
                # Position
                cloud_x.append(z)
                cloud_y.append(x)
                cloud_z.append(y)
                
                # Vector Direction (Raw momentum components)
                # Must match the coordinate mapping above! 
                # (px -> v, py -> w, pz -> u)
                cloud_u.append(pz) 
                cloud_v.append(px)
                cloud_w.append(py)
                
                # Color (Constant based on charge)
                if particle > 0:
                    cloud_c.append('tab:blue')
                else:
                    cloud_c.append('tab:orange')

            # Convert to numpy for plotting
            cx, cy, cz = np.array(cloud_x), np.array(cloud_y), np.array(cloud_z)
            cu, cv, cw = np.array(cloud_u), np.array(cloud_v), np.array(cloud_w)
            colors = np.array(cloud_c)
            
            # 1. Plot Spheres (The particles)
            # 'depthshade=True' gives them a 3D sphere look
            ax.scatter(cx, cy, cz, c=colors, s=15, alpha=0.8, depthshade=True, edgecolors='k', linewidth=0.1)
            
            # 2. Plot Arrows (The momentum)
            # normalize=False: Arrow length is proportional to vector magnitude (momentum)
            # length=0.03: Scaling factor. e.g., 100 GeV -> 3 units long
            ax.quiver(cx, cy, cz, cu, cv, cw, 
                      colors=colors, 
                      length=0.03,  
                      arrow_length_ratio=0.3, # Size of the arrowhead
                      linewidth=0.8, 
                      normalize=False)

    # ==========================================
    # 3. FINAL LAYOUT ADJUSTMENTS
    # ==========================================
    
    # Calculate limits to ensure 1:1:1 physical aspect ratio
    if sensitive_film_position is None and not 'sensitive_box' in detector: 
        z_lims = (31, -2)
    elif 'sensitive_box' in detector:
        z_lims = (85, -2)
    else:
        z_lims = (sensitive_film_position+1, -3)
             
    x_lims = (-1, 1)
    y_lims = (-1, 1)


    # Set Aspect Ratio
    box_z = abs(z_lims[0] - z_lims[1])
    box_x = abs(x_lims[0] - x_lims[1])
    box_y = abs(y_lims[0] - y_lims[1])
    ax.set_box_aspect((box_z, box_x, box_y))

    ax.set_xlim(z_lims)
    ax.set_ylim(x_lims)
    ax.set_zlim(y_lims)

    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if output_file:
        fig.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True, format='pdf')
        print(f'Fancy plot saved to {output_file}')
    
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot magnet geometry and muon tracks.')
    parser.add_argument('--azim', type=float, default=126, help='Azimuth angle for the 3D view')
    parser.add_argument('--elev', type=float, default=17, help='Elevation angle for the 3D view')
    args = parser.parse_args()
    phi = np.array([
        [0,115.50,50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 1.00, 1.00, 50.00, 50.00, 0.00, 0.00, 1.9],
        [10,225.38, 52.25, 78.36, 6.54, 9.37, 2.04, 40.36, 1.01, 1.05, 52.88, 82.20, 0.00, 0.00, 1.9],
        [10,274.43,52.47, 11.70, 41.59, 79.05, 2.19, 2.01, 1.03, 1.00, 54.15, 11.70, 0.00, 0.00, 1.9],
        [10,284.85, 33.05, 24.10, 55.13, 30.61, 90.36, 2.00, 1.00, 1.00, 33.05, 24.10, 0.11, 0.11, 1.9],
        [15,114.58, 5.62, 60.38, 31.37, 5.00, 2.00, 2.86, 1.00, 0.74, 5.59, 44.82, 0.00, 0.00, -1.9],
        [10,165.53, 90.85, 7.44, 137.45, 66.69, 10.09, 2.00, 0.88, 1.00, 80.16, 7.41, 0.00, 0.00, -1.9],
        [10,244.28,9.81, 47.60, 19.59, 164.19, 2.00, 2.05, 1.00, 0.89, 9.81, 42.16, 0.10, 0.10, -1.9]
    ])
    sens_plane = [{'dz': 0.01, 'dx': 4, 'dy': 6, 'position':82}]
    detector = get_design_from_params(params = phi,fSC_mag = False, simulate_fields=False, field_map_file = None, sensitive_film_params=None, add_cavern=False,add_target = False,  sensitive_decay_vessel=None, SND = False)
    N = 10
    px = np.random.normal(0,5,N)
    py = np.random.normal(0,5,N)
    pz = np.random.uniform(10,50,N)
    x = np.random.normal(0,0.2,N)
    y = np.random.normal(0,0.2,N)
    z = -1.5*np.ones(N)
    q = np.random.choice([1,-1],size=N)
    muons = np.column_stack((px, py, pz, x, y, z, q))
    plot_magnet(detector, muon_data=muons, azim=args.azim, elev=args.elev, sensitive_film_position=[82], plot_axis = False)