# sds_library/metal_kernels.py

EVALUATOR_KERNEL_CODE = """
#include <metal_stdlib>
using namespace metal;

float4 render_pixel_metal(
    int x,
    int y,
    const device float* shape_params,
    const device float4* shape_colors,
    int shapes_per_agent,
    float4 background_color
) {
    float4 final_color = background_color;

    for (int i = 0; i < shapes_per_agent; ++i) {
        int param_index = i * 7;
        int shape_type = int(shape_params[param_index]);

        bool is_inside = false;
        if (shape_type == 2) { // Triangle
            float p1_x = shape_params[param_index + 1];
            float p1_y = shape_params[param_index + 2];
            float p2_x = shape_params[param_index + 3];
            float p2_y = shape_params[param_index + 4];
            float p3_x = shape_params[param_index + 5];
            float p3_y = shape_params[param_index + 6];

            float d1 = (x - p3_x) * (p2_y - p3_y) - (p2_x - p3_x) * (y - p3_y);
            float d2 = (x - p1_x) * (p3_y - p1_y) - (p3_x - p1_x) * (y - p1_y);
            float c = (y - p1_y) * (p2_x - p1_x) - (x - p1_x) * (p2_y - p1_y);
            if (!((d1 < 0 || d2 < 0 || c < 0) && (d1 > 0 || d2 > 0 || c > 0))) {
                 is_inside = true;
            }
        }

        if (is_inside) {
            float4 fg_color = shape_colors[i];
            float4 bg_color = final_color;

            float fg_alpha = fg_color.a;
            final_color.rgb = fg_color.rgb * fg_alpha + bg_color.rgb * (1.0 - fg_alpha);
            final_color.a = fg_alpha + bg_color.a * (1.0 - fg_alpha);
        }
    }
    return final_color;
}

kernel void evaluate_population(
    const device float* population_params   [[buffer(0)]],
    const device float* population_colors   [[buffer(1)]],
    const device uchar4* target_image       [[buffer(2)]],
    const device int2* blocks               [[buffer(3)]],

    // --- MODIFIED: Changed to pointers for single values ---
    const device uint* block_size           [[buffer(4)]],
    const device uint* shapes_per_agent     [[buffer(5)]],
    const device uint* n_samples            [[buffer(6)]],
    const device uint* image_width          [[buffer(7)]],

    device float* out_scores                [[buffer(8)]],
    uint agent_id [[thread_position_in_grid]]
) {
    uint agent_param_start_index = agent_id * (*shapes_per_agent) * 7;
    uint agent_color_start_index = agent_id * (*shapes_per_agent);

    const device float* agent_shape_params = &population_params[agent_param_start_index];
    const device float4* agent_shape_colors_raw = (const device float4*)&population_colors[agent_color_start_index * 4];

    float4 background_color = float4(1.0, 1.0, 1.0, 1.0);
    // --- MODIFIED: Changed 'double' to 'float' ---
    float total_block_error = 0.0;

    for (uint i = 0; i < *n_samples; ++i) {
        int2 start = blocks[i];
        
        for (uint y_offset = 0; y_offset < *block_size; ++y_offset) {
            for (uint x_offset = 0; x_offset < *block_size; ++x_offset) {
                int x = start.x + x_offset;
                int y = start.y + y_offset;

                float4 predicted_color = render_pixel_metal(x, y, agent_shape_params, agent_shape_colors_raw, *shapes_per_agent, background_color);
                uchar4 actual_color_uchar = target_image[y * (*image_width) + x];
                float4 actual_color = float4(actual_color_uchar) / 255.0;

                total_block_error += fabs(predicted_color.r - actual_color.r);
                total_block_error += fabs(predicted_color.g - actual_color.g);
                total_block_error += fabs(predicted_color.b - actual_color.b);
                total_block_error += fabs(predicted_color.a - actual_color.a);
            }
        }
    }
    
    float num_pixels_checked = (*n_samples) * (*block_size) * (*block_size);
    out_scores[agent_id] = total_block_error / num_pixels_checked;
}
"""