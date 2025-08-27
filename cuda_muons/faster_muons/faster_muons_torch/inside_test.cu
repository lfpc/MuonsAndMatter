#include <cuda_runtime.h>
#include <stdio.h>

// Simple 2D vector struct for clarity
struct float2 {
    float x, y;
};

// Represents the geometry of a single ARB8 block on the GPU
struct ARB8_Data {
    // Vertices of the face at z = -dz (counter-clockwise order)
    float2 vertices_neg_z[4];
    // Vertices of the face at z = +dz (counter-clockwise order)
    float2 vertices_pos_z[4];
    // Z-center and half-length
    float z_center;
    float dz;
};

// --- The Core 'is_inside' Function ---
// This function would be called from within your CUDA kernel for each candidate block.
__device__ bool is_inside_arb8(const float3& particle_pos, const ARB8_Data& block) {
    // --- 1. Fast Z-Axis Check ---
    // Check if the particle is between the front and back Z-planes.
    if (fabsf(particle_pos.z - block.z_center) > block.dz) {
        return false; // Outside, no more checks needed.
    }

    // --- 2. XY Side-Planes Check ---

    // First, calculate the interpolation factor 'f' based on the particle's z-position.
    // f = 0 at the -dz face, f = 1 at the +dz face.
    float f = (particle_pos.z - (block.z_center - block.dz)) / (2.0f * block.dz);

    // Linearly interpolate to find the 4 vertices of the cross-section
    // at the particle's current z-height.
    float2 interpolated_verts[4];
    for (int i = 0; i < 4; ++i) {
        interpolated_verts[i].x = (1.0f - f) * block.vertices_neg_z[i].x + f * block.vertices_pos_z[i].x;
        interpolated_verts[i].y = (1.0f - f) * block.vertices_neg_z[i].y + f * block.vertices_pos_z[i].y;
    }

    // Now perform the 2D point-in-polygon test using the cross product.
    // We check if the particle's (x,y) is on the same side of all four edges.
    // The sign of the cross product tells us which side the point is on.
    // For a convex polygon with CCW vertices, the point is inside if all signs are positive.
    
    float signs[4];
    
    // Edge 0 -> 1
    float2 edge0 = {interpolated_verts[1].x - interpolated_verts[0].x, interpolated_verts[1].y - interpolated_verts[0].y};
    float2 p_vec0 = {particle_pos.x - interpolated_verts[0].x, particle_pos.y - interpolated_verts[0].y};
    signs[0] = edge0.x * p_vec0.y - edge0.y * p_vec0.x;

    // Edge 1 -> 2
    float2 edge1 = {interpolated_verts[2].x - interpolated_verts[1].x, interpolated_verts[2].y - interpolated_verts[1].y};
    float2 p_vec1 = {particle_pos.x - interpolated_verts[1].x, particle_pos.y - interpolated_verts[1].y};
    signs[1] = edge1.x * p_vec1.y - edge1.y * p_vec1.x;

    // Edge 2 -> 3
    float2 edge2 = {interpolated_verts[3].x - interpolated_verts[2].x, interpolated_verts[3].y - interpolated_verts[2].y};
    float2 p_vec2 = {particle_pos.x - interpolated_verts[2].x, particle_pos.y - interpolated_verts[2].y};
    signs[2] = edge2.x * p_vec2.y - edge2.y * p_vec2.x;

    // Edge 3 -> 0
    float2 edge3 = {interpolated_verts[0].x - interpolated_verts[3].x, interpolated_verts[0].y - interpolated_verts[3].y};
    float2 p_vec3 = {particle_pos.x - interpolated_verts[3].x, particle_pos.y - interpolated_verts[3].y};
    signs[3] = edge3.x * p_vec3.y - edge3.y * p_vec3.x;

    // The point is inside if all signs are the same (e.g., all >= 0).
    // This check is robust and handles points exactly on an edge.
    if ((signs[0] >= 0 && signs[1] >= 0 && signs[2] >= 0 && signs[3] >= 0) ||
        (signs[0] <= 0 && signs[1] <= 0 && signs[2] <= 0 && signs[3] <= 0)) {
        return true;
    }

    return false;
}

// --- Example Kernel to demonstrate usage ---
__global__ void check_inside_kernel(float3 particle, ARB8_Data block, bool* result) {
    *result = is_inside_arb8(particle, block);
}

int main() {
    // --- Setup Data on Host ---
    ARB8_Data h_block;
    h_block.z_center = 5.0f;
    h_block.dz = 5.0f; // Block extends from z=0 to z=10

    // A simple rectangular box for this example
    // Face at z=0 (-dz)
    h_block.vertices_neg_z[0] = {-1.0f, -1.0f};
    h_block.vertices_neg_z[1] = { 1.0f, -1.0f};
    h_block.vertices_neg_z[2] = { 1.0f,  1.0f};
    h_block.vertices_neg_z[3] = {-1.0f,  1.0f};
    // Face at z=10 (+dz)
    h_block.vertices_pos_z[0] = {-2.0f, -2.0f};
    h_block.vertices_pos_z[1] = { 2.0f, -2.0f};
    h_block.vertices_pos_z[2] = { 2.0f,  2.0f};
    h_block.vertices_pos_z[3] = {-2.0f,  2.0f};

    float3 particle_inside = {0.5f, 0.5f, 5.0f};
    float3 particle_outside = {3.0f, 3.0f, 5.0f};

    // --- Allocate Memory on Device ---
    ARB8_Data* d_block;
    bool* d_result;
    cudaMalloc(&d_block, sizeof(ARB8_Data));
    cudaMalloc(&d_result, sizeof(bool));

    // --- Copy Data to Device ---
    cudaMemcpy(d_block, &h_block, sizeof(ARB8_Data), cudaMemcpyHostToDevice);

    // --- Launch Kernel & Check ---
    bool h_result;
    
    // Test with inside particle
    check_inside_kernel<<<1, 1>>>(particle_inside, *d_block, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("Particle at (0.5, 0.5, 5.0) is inside: %s\n", h_result ? "true" : "false");

    // Test with outside particle
    check_inside_kernel<<<1, 1>>>(particle_outside, *d_block, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("Particle at (3.0, 3.0, 5.0) is inside: %s\n", h_result ? "true" : "false");

    // --- Free Memory ---
    cudaFree(d_block);
    cudaFree(d_result);

    return 0;
}
