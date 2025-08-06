#include <torch/extension.h>



// Forward declaration of the CUDA function
torch::Tensor propagate_muons_cuda(torch::Tensor muon_data, torch::Tensor loss_hists, torch::Tensor bin_widths, torch::Tensor bin_centers, int num_steps);

//torch::Tensor propagate_muons_cuda(torch::Tensor input1, torch::Tensor input2, int num_steps);
// Define a wrapper function for the propagate_muons operation
torch::Tensor propagate_muons(torch::Tensor muon_data, torch::Tensor loss_hists, torch::Tensor bin_widths, torch::Tensor bin_centers, int num_steps) {
    // Check that inputs are CUDA tensors
    if (!muon_data.is_cuda() || !loss_hists.is_cuda() || !bin_widths.is_cuda() || !bin_centers.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }

    // Call the CUDA implementation
    return propagate_muons_cuda(muon_data, loss_hists, bin_widths, bin_centers, num_steps);
}



// Forward declaration of the CUDA function
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
);

//torch::Tensor propagate_muons_with_alias_sampling_cuda(torch::Tensor input1, torch::Tensor input2, int num_steps);
// Define a wrapper function for the propagate_muons_with_alias_sampling operation
void propagate_muons_with_alias_sampling(
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
    // Check that inputs are CUDA tensors
    if (!muon_data_positions.is_cuda() ||
        !muon_data_momenta.is_cuda() ||
        !hist_2d_probability_table.is_cuda() ||
        !hist_2d_alias_table.is_cuda() ||
        !hist_2d_bin_centers_first_dim.is_cuda() ||
        !hist_2d_bin_centers_second_dim.is_cuda() ||
        !hist_2d_bin_widths_first_dim.is_cuda() ||
        !hist_2d_bin_widths_second_dim.is_cuda() ||
        !magnetic_field.is_cuda() ||
        magnetic_field_range.is_cuda() ||
        !arb8s.is_cuda() ||
        !hashed3d_arb8s.is_cuda() ||
        hashed3d_arb8s_range.is_cuda()
        )
    {
        throw std::runtime_error("All tensors except magnetic_field_range and hashed3d_arb8s_range must be CUDA tensors. magnetic_field_range and hashed3d_arb8s_range should be on the CPU.");
    }

    // Call the CUDA implementation
    propagate_muons_with_alias_sampling_cuda(
        muon_data_positions,
        muon_data_momenta,
        hist_2d_probability_table,
        hist_2d_alias_table,
        hist_2d_bin_centers_first_dim,
        hist_2d_bin_centers_second_dim,
        hist_2d_bin_widths_first_dim,
        hist_2d_bin_widths_second_dim,
        magnetic_field,
        magnetic_field_range,
        arb8s,
        hashed3d_arb8s,
        hashed3d_arb8s_range,
        kill_at,
        num_steps,
        step_length_fixed,
        seed
    );
}


torch::Tensor propagate_muons_rot_test_cuda( torch::Tensor muon_data,int num_steps );
torch::Tensor propagate_muons_rot_test(torch::Tensor muon_data, int num_steps) {
    // Check that inputs are CUDA tensors
    if (!muon_data.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }

    // Call the CUDA implementation
    return propagate_muons_rot_test_cuda(muon_data, num_steps);
}


void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor out);

// Define a wrapper function that can be called from Python for addition
void add(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    // Check that inputs are on the same CUDA device
    if (!x.is_cuda() || !y.is_cuda() || !out.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }
    add_cuda(x, y, out);
}
// Register the functions as PyTorch extensions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors (CUDA)");
    m.def("propagate_muons", &propagate_muons, "Propagate muons with two input tensors (CUDA)");
    m.def("propagate_muons_with_alias_sampling", &propagate_muons_with_alias_sampling, "Propagate muons with two input tensors (CUDA)");
    m.def("propagate_muons_rot_test", &propagate_muons_rot_test, "Propagate muons rot test");
}
