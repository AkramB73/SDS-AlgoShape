# sds_library/metal_kernels.py

EVALUATOR_KERNEL_CODE = """
#include <metal_stdlib>
using namespace metal;

// The helper function to render a pixel using a specific list of shape indices.
// This remains the same as our last version.
float4 render_pixel_from_indices(
    int x,
    int y,
    const device float* all_shape_params,
    const device float4* all_shape_colors,
    const device int* shape_indices_in_cell,
    int shape_count_in_cell,
    float4 background_color
) {
    float4 final_color = background_color;

    for (int i = 0; i < shape_count_in_cell; ++i) {
        int shape_index = shape_indices_in_cell[i];

        int param_index = shape_index * 7;
        int shape_type = int(all_shape_params[param_index]);

        bool is_inside = false;
        if (shape_type == 2) { // Triangle
            float p1_x = all_shape_params[param_index + 1];
            float p1_y = all_shape_params[param_index + 2];
            float p2_x = all_shape_params[param_index + 3];
            float p2_y = all_shape_params[param_index + 4];
            float p3_x = all_shape_params[param_index + 5];
            float p3_y = all_shape_params[param_index + 6];

            float d1 = (x - p3_x) * (p2_y - p3_y) - (p2_x - p3_x) * (y - p3_y);
            float d2 = (x - p1_x) * (p3_y - p1_y) - (p3_x - p1_x) * (y - p1_y);
            float c = (y - p1_y) * (p2_x - p1_x) - (x - p1_x) * (p2_y - p1_y);
            if (!((d1 < 0 || d2 < 0 || c < 0) && (d1 > 0 || d2 > 0 || c > 0))) {
                 is_inside = true;
            }
        }

        if (is_inside) {
            float4 fg_color = all_shape_colors[shape_index];
            float4 bg_color = final_color;

            float fg_alpha = fg_color.a;
            final_color.rgb = fg_color.rgb * fg_alpha + bg_color.rgb * (1.0 - fg_alpha);
            final_color.a = fg_alpha + bg_color.a * (1.0 - fg_alpha);
        }
    }
    return final_color;
}


// --- FINAL KERNEL: Massively Parallel and Grid-Aware ---
kernel void evaluate_all_agents_with_grids(
    // --- Mega Buffers containing data for ALL agents ---
    const device float* population_params,      // [[buffer(0)]]
    const device float* population_colors,      // [[buffer(1)]]
    const device int* mega_grid_indices,        // [[buffer(2)]]
    const device int2* mega_cell_offsets,       // [[buffer(3)]]
    const device int2* agent_grid_offsets,      // [[buffer(4)]] // Lookup for where each agent's grid data starts

    // Target image and sample locations
    const device uchar4* target_image,          // [[buffer(5)]]
    const device int2* blocks,                  // [[buffer(6)]]

    // Parameters
    const device uint* shapes_per_agent_ptr,    // [[buffer(7)]]
    const device uint* n_samples_ptr,           // [[buffer(8)]]
    const device uint* block_size_ptr,          // [[buffer(9)]]
    const device uint* image_width_ptr,         // [[buffer(10)]]
    const device uint* grid_size_ptr,           // [[buffer(11)]]
    const device float* cell_width_ptr,         // [[buffer(12)]]
    const device float* cell_height_ptr,        // [[buffer(13)]]

    // Output buffer for all pixel errors
    device float* out_pixel_errors,             // [[buffer(14)]]

    // System-generated unique ID for this thread
    uint thread_id [[thread_position_in_grid]]
) {
    // Dereference pointers once for efficiency
    uint shapes_per_agent = *shapes_per_agent_ptr;
    uint n_samples = *n_samples_ptr;
    uint block_size = *block_size_ptr;
    uint image_width = *image_width_ptr;
    uint grid_size = *grid_size_ptr;
    float cell_width = *cell_width_ptr;
    float cell_height = *cell_height_ptr;
    
    // --- 1. Determine which agent and pixel this thread is responsible for ---
    uint pixels_per_block = block_size * block_size;
    uint pixels_per_agent = n_samples * pixels_per_block;
    uint agent_id = thread_id / pixels_per_agent;
    uint pixel_index_in_agent = thread_id % pixels_per_agent;
    uint block_id = pixel_index_in_agent / pixels_per_block;
    uint pixel_index_in_block = pixel_index_in_agent % pixels_per_block;
    uint y_offset = pixel_index_in_block / block_size;
    uint x_offset = pixel_index_in_block % block_size;
    int2 block_start = blocks[block_id];
    int x = block_start.x + x_offset;
    int y = block_start.y + y_offset;

    // --- 2. Isolate this agent's shape data ---
    const device float* agent_shape_params = &population_params[agent_id * shapes_per_agent * 7];
    const device float4* agent_shape_colors = (const device float4*)&population_colors[agent_id * shapes_per_agent * 4];

    // --- 3. Find the grid cell and get the relevant shapes for THIS agent ---
    int2 agent_offset_info = agent_grid_offsets[agent_id];
    int agent_indices_start = agent_offset_info.x;
    int agent_cells_start = agent_offset_info.y;

    int col = (int)(x / cell_width);
    int row = (int)(y / cell_height);
    int cell_index_in_agent = row * grid_size + col;
    
    int2 cell_offset_info = mega_cell_offsets[agent_cells_start + cell_index_in_agent];
    int start_offset = cell_offset_info.x;
    int shape_count_in_cell = cell_offset_info.y;
    const device int* shape_indices_in_cell = &mega_grid_indices[agent_indices_start + start_offset];

    // --- 4. Render the pixel using ONLY the relevant shapes ---
    float4 background_color = float4(1.0, 1.0, 1.0, 1.0);
    float4 predicted_color = render_pixel_from_indices(x, y, 
        agent_shape_params, agent_shape_colors, 
        shape_indices_in_cell, shape_count_in_cell, 
        background_color);

    // --- 5. Calculate and write the error for this pixel ---
    uchar4 actual_color_uchar = target_image[y * image_width + x];
    float4 actual_color = float4(actual_color_uchar) / 255.0;
    
    float pixel_error = fabs(predicted_color.r - actual_color.r) +
                        fabs(predicted_color.g - actual_color.g) +
                        fabs(predicted_color.b - actual_color.b) +
                        fabs(predicted_color.a - actual_color.a);

    out_pixel_errors[thread_id] = pixel_error;
}
"""