from lib import magnet_simulations
import snoopy
import numpy as np
import matplotlib.pyplot as plt
import gmsh
        

params_HA = np.array([120.5, 50.0, 50.0, 119.0, 119.0, 2.0, 2.0, 1.0, 1.0, 50.0, 50.0, 0.0, 0.0, 0.])
params_SC = np.array([263.52 ,45.0, 45.0, 25.0, 25.0, 65.8, 50.65, 2.44, 3.29,107.83, 120.28, 0.0, 0.0, 3.2e6])
NI_0 = params_SC[-1] 



NIs = []
B_means = []
B_goal_list = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]


def get_dicts(params_HA, params_SC, B_goal, z_gap = 0.7):
    d_sc = magnet_simulations.get_magnet_params(params_SC, B_goal = B_goal, yoke_type='Mag2', z_gap = 0.)
    d_ha = magnet_simulations.get_magnet_params(params_HA, B_goal = 1.9, yoke_type='Mag1', z_gap = 0.)
    d_ha['Z_pos(m)'] = 0.0
    d_sc['Z_pos(m)'] = d_ha['Z_len(m)'] + z_gap
    d = {k: [d_ha[k], d_sc[k]] for k in d_ha.keys()}
    return d

def is_core(points, params, z_init):
    params = params/100
    core = (points[:,0]<params[1]) & (points[:,1]<params[3]) & (points[:,2]<(2*params[0]+z_init)) & (points[:,2]>z_init)
    return core

def simulate_magnet(d):
    points, B = magnet_simulations.run_fem(d).values()
    core = is_core(points, params_SC, z_init=d['Z_pos(m)'][1])
    assert core.sum() > 0, "No core points found"
    B_core = B[core]
    return B_core[:,1].mean()


d = get_dicts(params_HA, params_SC, B_goal=None)
B_core_mean = simulate_magnet(d)
points, B = magnet_simulations.run_fem(d).values()
core = (points[:,0]<params_SC[1]/100) & (points[:,1]<params_SC[3]/100)
assert core.sum() > 0, "No core points found"
B_core = B[core]
z_core = points[core, 2]
By_core = B_core[:, 1]
plt.figure(figsize=(8, 5))
plt.scatter(z_core, By_core, s=10, alpha=0.7)
plt.xlabel('z [m]')
plt.ylabel('B_y [T]')
plt.title('B_y vs z inside the core')
plt.grid(True)
plt.tight_layout()
plt.savefig("By_vs_z.png")
plt.close()

for B_goal in B_goal_list:
    d = get_dicts(params_HA, params_SC, B_goal)
    NI = d['NI(A)'][1]
    if np.isnan(NI):
        NI = 0
    B_core_mean = simulate_magnet(d)
    NIs.append(NI)
    B_means.append(B_core_mean)






fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First plot: B_goal vs NI
NIs_clipped = np.clip(NIs, None, 2e7)
axs[0].plot(B_goal_list, NIs_clipped, marker='o', label='NI (clipped at 20M)')
axs[0].axhline(NI_0, color='r', linestyle='--', label=f'NI_0 = {NI_0:.1e} A')
axs[0].set_xlabel('B_goal [T]')
axs[0].set_ylabel('NI [A-turns]')
axs[0].set_title('NI vs B_goal')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].grid(True, which='both', axis='y')

# Second plot: B_core vs B_goal
axs[1].plot(B_goal_list, B_means, marker='o', label='B_core mean')
axs[1].plot(B_goal_list, B_goal_list, 'k--', label='Identity')
axs[1].set_xlabel('B_goal [T]')
axs[1].set_ylabel('B_core mean [T]')
axs[1].set_title('B_core mean vs B_goal')
axs[1].legend()
axs[1].grid(True)

for NI in [1E4, 2E4, 5E4, 1E5, 2E5, 5E5, 1E6, 2E6, NI_0, 5E6, 1E7]:
    params_SC[-1] = NI
    d = get_dicts(params_HA, params_SC, B_goal=None)
    B_core_mean = simulate_magnet(d)
    B_means.append(B_core_mean)
    NIs.append(NI)

sorted_indices = np.argsort(NIs)
NIs = np.array(NIs)[sorted_indices]
B_means = np.array(B_means)[sorted_indices]

axs[2].plot(NIs,B_means, marker='o')
axs[2].axvline(NI_0, color='r', linestyle='--', label=f'NI_0 = {NI_0:.1e} A')
axs[2].set_ylabel('B_y [T]')
axs[2].set_xlabel('NI [A]')
axs[2].set_title('Average By vs NI')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("NI_vs_Bgoal.png")
plt.close()


print("NIs:", NIs)
print("B_means:", B_means)












