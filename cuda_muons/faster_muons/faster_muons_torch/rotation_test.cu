#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#define BLOCK_SIZE 1024

//
// // Helper function to compute the dot product of two vectors
// __device__ float dotProduct(const float a[3], const float b[3]) {
//     return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// }
//
// // Helper function to compute the cross product of two vectors
// __device__ void crossProduct(const float a[3], const float b[3], float result[3]) {
//     result[0] = a[1] * b[2] - a[2] * b[1];
//     result[1] = a[2] * b[0] - a[0] * b[2];
//     result[2] = a[0] * b[1] - a[1] * b[0];
// }
//
// // Helper function to compute the norm of a vector
// __device__ float norm(const float v[3]) {
//     return sqrt(dotProduct(v, v));
// }
//
// // Helper function to normalize a vector
// __device__ void normalize(const float v[3], float result[3]) {
//     float v_norm = norm(v);
// //     if (v_norm == 0) {
// //         printf("Error: Vector cannot be zero vector.\n");
// //         return; // Or handle as desired in CUDA (e.g., return default unit vector)
// //     }
//     result[0] = v[0] / v_norm;
//     result[1] = v[1] / v_norm;
//     result[2] = v[2] / v_norm;
// }
//
// // Function to rotate a vector delta_P to align with the direction of P
// __device__ void rotateVector(const float delta_P[3], const float P[3], float P_new[3]) {
//     // Normalize P to get the direction unit vector
//     float P_unit[3];
//     normalize(P, P_unit);
//
//     // Define the z-axis unit vector
//     float z_axis[3] = {0, 0, 1};
//
//     // Calculate the rotation axis (cross product of z-axis and P_unit)
//     float rotation_axis[3];
//     crossProduct(z_axis, P_unit, rotation_axis);
//     float rotation_axis_norm = norm(rotation_axis);
//
// //     // Check if rotation is needed
// //     if (rotation_axis_norm == 0) {
// //         // P is aligned with z-axis; no rotation is needed
// //         P_new[0] = P[0] + delta_P[0];
// //         P_new[1] = P[1] + delta_P[1];
// //         P_new[2] = P[2] + delta_P[2];
// //         return;
// //     }
//
//     // Normalize the rotation axis
//     normalize(rotation_axis, rotation_axis);
//
//     // Calculate the rotation angle
//     float cos_theta = dotProduct(z_axis, P_unit);
//     float theta = acos(cos_theta);
//
//     // Construct the rotation matrix using Rodrigues' rotation formula
//     float K[3][3] = {
//         {0, -rotation_axis[2], rotation_axis[1]},
//         {rotation_axis[2], 0, -rotation_axis[0]},
//         {-rotation_axis[1], rotation_axis[0], 0}
//     };
//
//     float R[3][3];
//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 3; ++j) {
//             R[i][j] = (i == j ? 1 : 0) +
//                       sin(theta) * K[i][j] +
//                       (1 - cos_theta) * (K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j]);
//         }
//     }
//
//     // Rotate delta_P using the rotation matrix
//     float delta_P_rotated[3] = {
//         R[0][0] * delta_P[0] + R[0][1] * delta_P[1] + R[0][2] * delta_P[2],
//         R[1][0] * delta_P[0] + R[1][1] * delta_P[1] + R[1][2] * delta_P[2],
//         R[2][0] * delta_P[0] + R[2][1] * delta_P[1] + R[2][2] * delta_P[2]
//     };
//
//     // Update P by adding the rotated delta_P to get the new position
//     P_new[0] = P[0] + delta_P_rotated[0];
//     P_new[1] = P[1] + delta_P_rotated[1];
//     P_new[2] = P[2] + delta_P_rotated[2];
// }
//
// __global__ void propagate_muons_rot_test_kernel(
//     float* muon_data,  // Input tensor holding the 3D positions of muons
//     int N,             // Number of muons
//     int num_steps      // Number of propagation steps
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (idx >= N) return;
//
//     // Initialize random number generator for each thread
//     curandState state;
//     curand_init(1234, idx, 0, &state);  // Use idx as seed for unique randomness
//
//     // Offset in muon_data for this muon's 3D position (assuming xyz layout)
//     int offset = idx * 3;
//
//     // Perform num_steps random walk for this muon
//     for (int step = 0; step < num_steps; ++step) {
//         // Generate a random 3D vector
//         float rand_vec[3];
//         float rand_vec_2[3];
//
//         rand_vec[0] = curand_uniform(&state) - 0.5f + 5;
//         rand_vec[1] = curand_uniform(&state) - 0.5f + 5;
//         rand_vec[2] = curand_uniform(&state) - 0.5f + 5;
//
//         rand_vec_2[0] = curand_uniform(&state) - 0.5f + 5;
//         rand_vec_2[1] = curand_uniform(&state) - 0.5f + 5;
//         rand_vec_2[2] = curand_uniform(&state) - 0.5f + 5;
//
//         rotateVector(rand_vec, muon_data + offset, rand_vec_2);
//
//         // Add the random vector to the current position in muon_data
//         muon_data[offset]     += rand_vec_2[0];
//         muon_data[offset + 1] += rand_vec_2[1];
//         muon_data[offset + 2] += rand_vec_2[2];
//     }
// }



torch::Tensor propagate_muons_rot_test_cuda(
    torch::Tensor muon_data,
    int num_steps  // Number of propagation steps
) {
    const auto N = muon_data.size(0);

    // Allocate output tensor
    auto muon_data_output = torch::empty_like(muon_data);

    // Define grid and block sizes
    const int threads_per_block = BLOCK_SIZE;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

//     // Launch the kernel
//     propagate_muons_rot_test_kernel<<<num_blocks, threads_per_block>>>(
//         muon_data_output.data_ptr<float>(),  // Pass the output tensor's data pointer
//         N,                                   // Number of muons
//         num_steps                            // Number of propagation steps
//     );


    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    return muon_data_output;
}
