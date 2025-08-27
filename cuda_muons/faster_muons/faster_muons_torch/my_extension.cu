#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>


// #define MUON_MASS 0.1056583755f  // GeV/c²
// #define c_speed_of_light 299792458.0f  // Speed of light in m/s
// #define e_charge 1.602176634e-19f  // Elementary charge in Coulombs
// // Conversion factor: 1 GeV/c in SI momentum units (kg·m/s)
// // #define GeV_over_c_to_SI  1.602176634e-10 / c_speed_of_light  // ≈ 5.3443e-19 kg·m/s
// #define GeV_over_c_to_SI  5.3443e-19f  // ≈ 5.3443e-19 kg·m/s

__global__ void add_kernel(float *x, float *y, float *out, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = x[index] + y[index];
    }
}

void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    int size = x.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size);
}


#define BLOCK_SIZE 256

__global__ void propagate_muons_kernel(
    float* muon_data,        // Input tensor
    float* muon_data_output, // Output tensor
    float* loss_hists,       // CDF histogram data
    float* bin_widths,       // Width of bins in histograms
    float* bin_centers,      // Center of bins in histograms
    int N,                   // Number of muons
    int M,                   // Number of histograms
    int H,                   // Number of bins in each histogram
    int num_steps            // Number of propagation steps
) {
//     extern __shared__ float loss_hists_shared[];
//
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//     // Copy global data to shared memory - usually only a subset of threads are used
//     if (idx < M*H) {
//         sharedData[idx] = globalData[idx];
//     }
//
//     // Synchronize to ensure shared memory is fully populated
//     __syncthreads();


    if (idx >= N) return;

    // Initialize random number generator for each thread
    curandState state;
    curand_init(1234, idx, 0, &state);  // Use idx as seed for unique randomness

    // Initialize output value for this muon
    float muon_value = muon_data[idx];

    // Loop over propagation steps
    for (int step = 0; step < num_steps; ++step) {
        // 1. Select a random histogram
        int hist_idx = curand(&state) % M;

        // 2. Generate a uniform random number in [0, 1] for sampling from CDF
        float rand_value = curand_uniform(&state);

        // Linear version
//         int start = hist_idx * H;
//         int end = start + H - 1;
//         int bin_idx = start;
//
//         while (bin_idx <= end) {
//             if (loss_hists[bin_idx] >= rand_value) {
//                 break; // Found the element
//             }
//             bin_idx++;
//         }


        // Binary search version

        // 3. Perform binary search in the selected histogram's CDF
        int start = hist_idx * H;
        int end = start + H - 1;
        int left = start;
        int right = end;
        while (left < right) {
            int mid = (left + right) / 2;
            if (loss_hists[mid] < rand_value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // 4. Retrieve the bin value and add it to muon_value
        int bin_idx = left;

        float bin_value = bin_centers[bin_idx]; // Initialize bin_value at the bin center
        float bin_jitter = (curand_uniform(&state) - 0.5f) * bin_widths[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
        bin_value += bin_jitter; // Apply jitter to the bin value

        muon_value += bin_value;
    }

    // Store the final result after all steps
    muon_data_output[idx] = muon_value;
}


torch::Tensor propagate_muons_cuda(
    torch::Tensor muon_data,
    torch::Tensor loss_hists,
    torch::Tensor bin_widths,
    torch::Tensor bin_centers,
    int num_steps  // Number of propagation steps
) {
    const auto N = muon_data.size(0);
    const auto M = loss_hists.size(0);
    const auto H = loss_hists.size(1);

    // Allocate output tensor
    auto muon_data_output = torch::empty_like(muon_data);

    // Define grid and block sizes
    const int threads_per_block = BLOCK_SIZE;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    propagate_muons_kernel<<<num_blocks, threads_per_block>>>(
        muon_data.data_ptr<float>(),
        muon_data_output.data_ptr<float>(),
        loss_hists.data_ptr<float>(),
        bin_widths.data_ptr<float>(),
        bin_centers.data_ptr<float>(),
        N, M, H, num_steps
    );

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    return muon_data_output;
}

void load_arb8s_from_tensor(const at::Tensor& arb8s_tensor, const at::Tensor& z_centers, const at::Tensor& dzs, std::vector<ARB8_Data>& out) {
    int N = arb8s_tensor.size(0);
    auto arb8s_acc = arb8s_tensor.accessor<float, 3>();
    auto z_centers_acc = z_centers.accessor<float, 1>();
    auto dzs_acc = dzs.accessor<float, 1>();
    out.resize(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i].vertices_neg_z[j].x = arb8s_acc[i][j][0];
            out[i].vertices_neg_z[j].y = arb8s_acc[i][j][1];
            out[i].vertices_pos_z[j].x = arb8s_acc[i][j+4][0];
            out[i].vertices_pos_z[j].y = arb8s_acc[i][j+4][1];
        }
        out[i].z_center = z_centers_acc[i];
        out[i].dz = dzs_acc[i];
    }
}

__device__ int get_first_bin(int num) {
    // Clip num to be within [10, 300]
    if (num < 10) {
        num = 10;
    } else if (num > 300) {
        num = 300;
    }

    // Calculate the range index
    int index = (num - 10) / 5;
    return index;
}


// Helper function to compute the dot product of two vectors
__device__ float dotProduct(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Helper function to compute the cross product of two vectors
__device__ void crossProduct(const float a[3], const float b[3], float result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

// Helper function to compute the norm of a vector
__device__ float norm(const float v[3]) {
    return sqrt(dotProduct(v, v));
}

// Helper function to normalize a vector
__device__ void normalize(const float v[3], float result[3]) {
    float v_norm = norm(v);
//     if (v_norm == 0) {
//         printf("Error: Vector cannot be zero vector.\n");
//         return; // Or handle as desired in CUDA (e.g., return default unit vector)
//     }
    result[0] = v[0] / v_norm;
    result[1] = v[1] / v_norm;
    result[2] = v[2] / v_norm;
}

// Constants
const float charge = 1.0f;         // Adjust based on the particle's charge


// Function to rotate a vector delta_P to align with the direction of P
__device__ void rotateVector(const float P_unit[3], const float delta_P[3], const float P[3], float P_new[3]) {
    // Define the z-axis unit vector
    float z_axis[3] = {0, 0, 1};

    // Calculate the rotation axis (cross product of z-axis and P_unit)
    float rotation_axis[3];
    crossProduct(z_axis, P_unit, rotation_axis);
    float rotation_axis_norm = norm(rotation_axis);

     // Check if rotation is needed
     if (rotation_axis_norm == 0) {
         // P is aligned with z-axis; no rotation is needed
         P_new[0] = P[0] + delta_P[0];
         P_new[1] = P[1] + delta_P[1];
         P_new[2] = P[2] + delta_P[2];
         return;
     }

    // Normalize the rotation axis
    normalize(rotation_axis, rotation_axis);

    // Calculate the rotation angle
    float cos_theta = dotProduct(z_axis, P_unit);
    float theta = acos(cos_theta);

    // Construct the rotation matrix using Rodrigues' rotation formula
    float K[3][3] = {
            {0, -rotation_axis[2], rotation_axis[1]},
            {rotation_axis[2], 0, -rotation_axis[0]},
            {-rotation_axis[1], rotation_axis[0], 0}
    };

    float R[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R[i][j] = (i == j ? 1 : 0) +
                      sin(theta) * K[i][j] +
                      (1 - cos_theta) * (K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j]);
        }
    }

    // Rotate delta_P using the rotation matrix
    float delta_P_rotated[3] = {
            R[0][0] * delta_P[0] + R[0][1] * delta_P[1] + R[0][2] * delta_P[2],
            R[1][0] * delta_P[0] + R[1][1] * delta_P[1] + R[1][2] * delta_P[2],
            R[2][0] * delta_P[0] + R[2][1] * delta_P[1] + R[2][2] * delta_P[2]
    };


    // Update P by adding the rotated delta_P to get the new position
    P_new[0] = P[0] + delta_P_rotated[0];
    P_new[1] = P[1] + delta_P_rotated[1];
    P_new[2] = P[2] + delta_P_rotated[2];
}

// For device and host compatibility.
__host__ __device__ inline float3 getFieldAt(const float *field,
                                              const float x, const float y, const float z,
                                              const int binsX, const int binsY, const int binsZ,
                                              const float startX, const float endX,
                                              const float startY, const float endY,
                                              const float startZ, const float endZ)
{
    if (x < startX || x > endX || y < startY || y > endY || z < startZ || z > endZ) {
        float3 nullField = {0.0f, 0.0f, 0.0f};
        return nullField;
    }
    // Normalize the physical coordinate into [0, 1].
    float normX = (x - startX) / (endX - startX);
    float normY = (y - startY) / (endY - startY);
    float normZ = (z - startZ) / (endZ - startZ);

    // Map the normalized coordinate to a bin index.
    // Multiplying by binsX (or binsY, binsZ) maps [0,1] to [0, binsX].
    // We then clamp the index to the valid range.
    int i = normX * binsX;
    int j = normY * binsY;
    int k = normZ * binsZ;

    if(i >= binsX) i = binsX - 1;
    if(j >= binsY) j = binsY - 1;
    if(k >= binsZ) k = binsZ - 1;

    // Compute the flat array index.
    // Each field point consists of 3 floats.
    int index = ((k * binsY * binsX) + (j * binsX) + i) * 3;

    // Create a float3 with the field values.
    float3 fieldVal;
    fieldVal.x = field[index];
    fieldVal.y = field[index + 1];
    fieldVal.z = field[index + 2];

    return fieldVal;
}


__constant__ float MUON_MASS = 0.1056583755f;  // GeV/c²
__constant__ float c_speed_of_light = 299792458.0f;  // Speed of light in m/s
__constant__ float e_charge = 1.602176634e-19f;  // Elementary charge in Coulombs
__constant__ float GeV_over_c_to_SI  = 5.3443e-19f;  // ≈ 5.3443e-19 kg·m/s

__device__ void derivative(float* state, float charge, const float* B, float* deriv,
              const int fieldBinsX, const int fieldBinsY, const int fieldBinsZ,
              const float fieldStartX, const float fieldEndX,
              const float fieldStartY, const float fieldEndY,
              const float fieldStartZ, const float fieldEndZ,
              const float p_mag, const float energy) {
    // Unpack the state array
    float x  = state[0];
    float y  = state[1];
    float z  = state[2];
    float px = state[3];
    float py = state[4];
    float pz = state[5];

    // Calculate the magnitude of the momentum and the energy
//     float p_mag  = sqrt(px * px + py * py + pz * pz);
//     float energy = sqrt(p_mag * p_mag + MUON_MASS * MUON_MASS);


//     printf("p_mag = %f, energy = %f\n", p_mag, energy);

    // Velocity components (v = p/E * c)

    float vx = (px / energy) * c_speed_of_light;
    float vy = (py / energy) * c_speed_of_light;
    float vz = (pz / energy) * c_speed_of_light;

    // Multiply the charge by the elementary charge (e_charge)
    float q = charge * e_charge;

    float3 B_ = getFieldAt(B, x, y, z,
            fieldBinsX, fieldBinsY, fieldBinsZ,
            fieldStartX, fieldEndX,
            fieldStartY, fieldEndY,
            fieldStartZ, fieldEndZ
        );

    // Lorentz force: dp/dt = q * (v x B) scaled by the conversion factor.
    float dpx = q * (vy * B_.z - vz * B_.y) / GeV_over_c_to_SI;
    float dpy = q * (vz * B_.x - vx * B_.z) / GeV_over_c_to_SI;
    float dpz = q * (vx * B_.y - vy * B_.x) / GeV_over_c_to_SI;

    // Pack the derivatives into the output array
    deriv[0] = vx;
    deriv[1] = vy;
    deriv[2] = vz;
    deriv[3] = dpx;
    deriv[4] = dpy;
    deriv[5] = dpz;

}

__device__ void rk4_step(float pos[3], float mom[3],
              float charge, float step_length_fixed,
              const float* B,
              float new_pos[3], float new_mom[3],
              const int fieldBinsX, const int fieldBinsY, const int fieldBinsZ,
              const float fieldStartX, const float fieldEndX,
              const float fieldStartY, const float fieldEndY,
              const float fieldStartZ, const float fieldEndZ)
{
    // Combine position and momentum into a single state vector.
    float state[6] = { pos[0], pos[1], pos[2], mom[0], mom[1], mom[2] };

    // Compute the magnitude of the momentum.
    float p_mag = sqrt(mom[0]*mom[0] + mom[1]*mom[1] + mom[2]*mom[2]);

//     float p_mag  = sqrt(px * px + py * py + pz * pz);
    float energy = sqrt(p_mag * p_mag + MUON_MASS * MUON_MASS);

    // Calculate the energy: E^2 = p^2 + m^2
//     float energy = sqrt(p_mag*p_mag + MUON_MASS*MUON_MASS);

    // Determine the velocity magnitude: v = (p/E) * c (for ultra-relativistic particles, v ~ c)
    float v_mag = (p_mag / energy) * c_speed_of_light;

    // Calculate time step dt = step_length / v
    float dt = step_length_fixed / v_mag;

    // Allocate arrays for RK4 slopes.
    float k1[6], k2[6],
          k3[6], k4[6];
    float temp[6];

//     // First RK4 step: k1
    derivative(state, charge, B, k1,
              fieldBinsX, fieldBinsY, fieldBinsZ,
              fieldStartX, fieldEndX,
              fieldStartY, fieldEndY,
              fieldStartZ, fieldEndZ,
              p_mag, energy);

    // Second RK4 step: k2
    for (int i = 0; i < 6; i++) {
        temp[i] = state[i] + 0.5 * dt * k1[i];
    }

    derivative(temp, charge, B, k2,
              fieldBinsX, fieldBinsY, fieldBinsZ,
              fieldStartX, fieldEndX,
              fieldStartY, fieldEndY,
              fieldStartZ, fieldEndZ,
              p_mag, energy);


    // Third RK4 step: k3
    for (int i = 0; i < 6; i++) {
        temp[i] = state[i] + 0.5 * dt * k2[i];
    }
    derivative(temp, charge, B, k3,
              fieldBinsX, fieldBinsY, fieldBinsZ,
              fieldStartX, fieldEndX,
              fieldStartY, fieldEndY,
              fieldStartZ, fieldEndZ,
              p_mag, energy);

    // Fourth RK4 step: k4
    for (int i = 0; i < 6; i++) {
        temp[i] = state[i] + dt * k3[i];
    }

    derivative(temp, charge, B, k4,
              fieldBinsX, fieldBinsY, fieldBinsZ,
              fieldStartX, fieldEndX,
              fieldStartY, fieldEndY,
              fieldStartZ, fieldEndZ,
              p_mag, energy);



    // Combine the intermediate slopes to compute the new state.
    float new_state[6];
    for (int i = 0; i < 6; i++) {
        new_state[i] = state[i]+ dt/6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }

    // Write the updated state into the output arrays.
    // The first three elements are position, and the last three are momentum.
    new_pos[0] = new_state[0];
    new_pos[1] = new_state[1];
    new_pos[2] = new_state[2];

    new_mom[0] = new_state[3];
    new_mom[1] = new_state[4];
    new_mom[2] = new_state[5];
}


enum MaterialType { MATERIAL_IRON, MATERIAL_CONCRETE, MATERIAL_AIR };
struct float2 {
    float x, y;
};
struct ARB8_Data {
    float2 vertices_neg_z[4];
    float2 vertices_pos_z[4];
    float z_center;
    float dz;
};

__device__ bool is_inside_arb8(const float3& particle_pos, const ARB8_Data& block) {
    if (fabsf(particle_pos.z - block.z_center) > block.dz) {
        return false;
    }
    float f = (particle_pos.z - (block.z_center - block.dz)) / (2.0f * block.dz);
    float2 interpolated_verts[4];
    for (int i = 0; i < 4; ++i) {
        interpolated_verts[i].x = (1.0f - f) * block.vertices_neg_z[i].x + f * block.vertices_pos_z[i].x;
        interpolated_verts[i].y = (1.0f - f) * block.vertices_neg_z[i].y + f * block.vertices_pos_z[i].y;
    }
    float signs[4];
    float2 edge, p_vec;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        edge.x = interpolated_verts[j].x - interpolated_verts[i].x;
        edge.y = interpolated_verts[j].y - interpolated_verts[i].y;
        p_vec.x = particle_pos.x - interpolated_verts[i].x;
        p_vec.y = particle_pos.y - interpolated_verts[i].y;
        signs[i] = edge.x * p_vec.y - edge.y * p_vec.x;
    }
    if ((signs[0] >= 0 && signs[1] >= 0 && signs[2] >= 0 && signs[3] >= 0) ||
        (signs[0] <= 0 && signs[1] <= 0 && signs[2] <= 0 && signs[3] <= 0)) {
        return true;
    }
    return false;
}
__device__ MaterialType get_material(float x, float y, float z) {
    // Example: hardcoded ARB8 block (replace with your actual block data or pass as argument)
    ARB8_Data block;
    block.z_center = 5.0f;
    block.dz = 5.0f;
    block.vertices_neg_z[0] = {-1.0f, -1.0f};
    block.vertices_neg_z[1] = { 1.0f, -1.0f};
    block.vertices_neg_z[2] = { 1.0f,  1.0f};
    block.vertices_neg_z[3] = {-1.0f,  1.0f};
    block.vertices_pos_z[0] = {-2.0f, -2.0f};
    block.vertices_pos_z[1] = { 2.0f, -2.0f};
    block.vertices_pos_z[2] = { 2.0f,  2.0f};
    block.vertices_pos_z[3] = {-2.0f,  2.0f};

    float3 pos = {x, y, z};
    if (is_inside_arb8(pos, block)) {
        return MATERIAL_IRON;
    }
    return MATERIAL_AIR;
}

__global__ void cuda_test_propagate_muons_k(float* muon_data_positions,
                               float* muon_data_momenta,
                               const float* hist_2d_probability_table,
                               const int* hist_2d_alias_table,
                               const float* hist_2d_bin_centers_first_dim,
                               const float* hist_2d_bin_centers_second_dim,
                               const float* hist_2d_bin_widths_first_dim,
                               const float* hist_2d_bin_widths_second_dim,
                               const float* magnetic_field,
                               const float kill_at,
                               const int N,
                               const int H_2d,
                               const int fieldBinsX, const int fieldBinsY, const int fieldBinsZ,
                               const float fieldStartX, const float fieldEndX,
                               const float fieldStartY, const float fieldEndY,
                               const float fieldStartZ, const float fieldEndZ,
                               int num_steps,
                               float step_length_fixed,
                               int seed)
                               {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;


    // Initialize random number generator for each thread
    curandState state;
    curand_init(seed, idx, 0, &state);  // Use idx as seed for unique randomness

    int offset = idx * 3;
    float delta_P[3] = {0,0,-1};
    float output[3] = {0, 0, 0};

//     float *muon_data_momenta_this = muon_data_momenta + offset;
//     float *muon_data_positions_this = muon_data_positions + offset;

    float muon_data_momenta_this_cached[3] = {muon_data_momenta[offset+0], muon_data_momenta[offset+1], muon_data_momenta[offset+2]};
    float muon_data_positions_this_cached[3] = {muon_data_positions[offset+0], muon_data_positions[offset+1], muon_data_positions[offset+2]};

    for (int step = 0; step < num_steps; step++) {
        float mag_P = norm(muon_data_momenta_this_cached);

        // Normalize P to get the direction unit vector
        float P_unit[3];
        normalize(muon_data_momenta_this_cached, P_unit);
        int hist_idx = get_first_bin((int)mag_P);

        if (hist_idx==-1)
            break;

        if (kill_at != -1 and mag_P < kill_at)
            break;

        // 2. Generate a uniform random number in [0, 1] for sampling from CDF
        float rand_value = curand_uniform(&state);
        // Get material at current position (dummy for now)
        MaterialType material = get_material(
            muon_data_positions_this_cached[0],
            muon_data_positions_this_cached[1],
            muon_data_positions_this_cached[2]
        );

        float delta = 0.0f;
        float delta_second_dim = 0.0f;
        if (material == MATERIAL_IRON) {
            int bin_idx;
            int tbin = curand(&state) % H_2d;
            
            if (rand_value < hist_2d_probability_table[tbin+hist_idx*H_2d])
                bin_idx = tbin;
            else
                bin_idx =  hist_2d_alias_table[tbin+hist_idx*H_2d];
            // 3. Retrieve the bin value and add it to muon_valu
            float bin_value_first_dim = hist_2d_bin_centers_first_dim[bin_idx]; // Initialize bin_value at the bin center
            float bin_jitter_first_dim = (curand_uniform(&state) - 0.5f) * hist_2d_bin_widths_first_dim[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
            bin_value_first_dim += bin_jitter_first_dim; // Apply jitter to the bin value

            float bin_value_second_dim = hist_2d_bin_centers_second_dim[bin_idx]; // Initialize bin_value at the bin center
            float bin_jitter_second_dim = (curand_uniform(&state) - 0.5f) * hist_2d_bin_widths_second_dim[bin_idx]; // Calculate jitter in range [-bin_width/2, bin_width/2]
            bin_value_second_dim += bin_jitter_second_dim; // Apply jitter to the bin value

            delta = mag_P * exp(bin_value_first_dim);
            delta_second_dim = mag_P * exp(bin_value_second_dim);
        }

        float phi = curand_uniform(&state) * 2 * M_PI;

        // Convert polar coordinates to Cartesian coordinates
        float x = delta_second_dim * cos(phi);
        float y = delta_second_dim * sin(phi);

        delta_P[0] = x;
        delta_P[1] = -y;
        delta_P[2] = -delta;

        rotateVector(P_unit, delta_P, muon_data_momenta_this_cached, output);
        // works till here

        muon_data_momenta_this_cached[0] = output[0];
        muon_data_momenta_this_cached[1] = output[1];
        muon_data_momenta_this_cached[2] = output[2];

        rk4_step(muon_data_positions_this_cached, muon_data_momenta_this_cached,
                      +1, step_length_fixed,
                      magnetic_field,
                      muon_data_positions_this_cached, muon_data_momenta_this_cached,
                      fieldBinsX, fieldBinsY, fieldBinsZ,
                      fieldStartX, fieldEndX,
                      fieldStartY, fieldEndY,
                      fieldStartZ, fieldEndZ);
    }

    muon_data_momenta[offset+0] = muon_data_momenta_this_cached[0];
    muon_data_momenta[offset+1] = muon_data_momenta_this_cached[1];
    muon_data_momenta[offset+2] = muon_data_momenta_this_cached[2];

    muon_data_positions[offset+0] = muon_data_positions_this_cached[0];
    muon_data_positions[offset+1] = muon_data_positions_this_cached[1];
    muon_data_positions[offset+2] = muon_data_positions_this_cached[2];
}


void propagate_muons_with_alias_sampling_cuda(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor hist_2d_probability_table,
    torch::Tensor hist_2d_alias_table,
    torch::Tensor hist_2d_bin_centers_first_dim,
    torch::Tensor hist_2d_bin_centers_second_dim,
    torch::Tensor hist_2d_bin_widths_first_dim,
    torch::Tensor hist_2d_bin_widths_second_dim,
    torch::Tensor magnetic_field,
    torch::Tensor magnetic_field_range,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s,
    torch::Tensor hashed3d_arb8s_range,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
) {
    TORCH_CHECK(arb8s.size(1) == 5, "Expected the second dimension of arb8s to be 5, but got ", arb8s.size(1));
    TORCH_CHECK(magnetic_field_range.size(1) == 6, "Expected the second dimension of magnetic_field_range to be 6, but got ", magnetic_field_range.size(1));
    TORCH_CHECK(hashed3d_arb8s_range.size(1) == 6, "Expected the second dimension of hashed3d_arb8s_range to be 6, but got ", hashed3d_arb8s_range.size(1));

    const auto N = muon_data_positions.size(0);
    const auto H_2D = hist_2d_probability_table.size(1);

    const auto nx_mag_field = magnetic_field.size(0);
    const auto ny_mag_field = magnetic_field.size(1);
    const auto nz_mag_field = magnetic_field.size(2);

    // Define grid and block sizes
    const int threads_per_block = BLOCK_SIZE;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    float* data_ptr = magnetic_field_range.data_ptr<float>();
    float* data_ptr_arbs = hashed3d_arb8s_range.data_ptr<float>();

    float rangeStartX = data_ptr[0];
    float rangeEndX = data_ptr[1];
    float rangeStartY = data_ptr[2];
    float rangeEndY = data_ptr[3];
    float rangeStartZ = data_ptr[4];
    float rangeEndZ = data_ptr[5];

    float rangeStartXArbs = data_ptr_arbs[0];
    float rangeEndXArbs = data_ptr_arbs[1];
    float rangeStartYArbs = data_ptr_arbs[2];
    float rangeEndYArbs = data_ptr_arbs[3];
    float rangeStartZArbs = data_ptr_arbs[4];
    float rangeEndZArbs = data_ptr_arbs[5];

    // Prepare ARB8s
    std::vector<ARB8_Data> arb8s_host;
    load_arb8s_from_tensor(arb8s, arb8s.select(2,2), arb8s.select(2,3), arb8s_host); // Adjust if z_center/dz are in separate tensors

    ARB8_Data* arb8s_device;
    cudaMalloc(&arb8s_device, arb8s_host.size() * sizeof(ARB8_Data));
    cudaMemcpy(arb8s_device, arb8s_host.data(), arb8s_host.size() * sizeof(ARB8_Data), cudaMemcpyHostToDevice);


    #define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)


    // Launch the kernel with raw pointers
    cuda_test_propagate_muons_k<<<num_blocks, threads_per_block>>>(
        muon_data_positions.data_ptr<float>(),
        muon_data_momenta.data_ptr<float>(),
        hist_2d_probability_table.data_ptr<float>(),
        hist_2d_alias_table.data_ptr<int>(),
        hist_2d_bin_centers_first_dim.data_ptr<float>(),
        hist_2d_bin_centers_second_dim.data_ptr<float>(),
        hist_2d_bin_widths_first_dim.data_ptr<float>(),
        hist_2d_bin_widths_second_dim.data_ptr<float>(),
        magnetic_field.data_ptr<float>(),
        kill_at,
        N,
        H_2D,
        nx_mag_field,
        ny_mag_field,
        nz_mag_field,
        rangeStartX, rangeEndX,
        rangeStartY, rangeEndY,
        rangeStartZ, rangeEndZ,
        num_steps,
        step_length_fixed,
        seed
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(arb8s_device);

//     // Synchronize to ensure kernel completion
//     cudaDeviceSynchronize();
}