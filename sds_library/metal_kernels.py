EVALUATOR_KERNEL_CODE = """
#include <metal_stdlib>
using namespace metal;

float4 render_pixel_metal(
    int x, int y, const device float* shape_params,
    const device float4* shape_colors, int shapes_per_agent,
    float4 background_color
) {
    float4 final_color = background_color;
    for (int i = 0; i < shapes_per_agent; ++i) {
        int param_index = i * 7;
        int shape_type = int(shape_params[param_index]);
        bool is_inside = false;
        if (shape_type == 2) { // Triangle
            float p1_x = shape_params[param_index + 1]; float p1_y = shape_params[param_index + 2];
            float p2_x = shape_params[param_index + 3]; float p2_y = shape_params[param_index + 4];
            float p3_x = shape_params[param_index + 5]; float p3_y = shape_params[param_index + 6];
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

kernel void evaluate_pixel_errors(
    const device float* population_params, const device float* population_colors,
    const device uchar4* target_image, const device int2* blocks,
    const device uint* shapes_per_agent_ptr, const device uint* n_samples_ptr,
    const device uint* block_size_ptr, const device uint* image_width_ptr,
    device float* out_pixel_errors, uint thread_id [[thread_position_in_grid]]
) {
    uint shapes_per_agent = *shapes_per_agent_ptr;
    uint n_samples = *n_samples_ptr;
    uint block_size = *block_size_ptr;
    uint image_width = *image_width_ptr;
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
    const device float* agent_shape_params = &population_params[agent_id * shapes_per_agent * 7];
    const device float4* agent_shape_colors = (const device float4*)&population_colors[agent_id * shapes_per_agent * 4];
    float4 background_color = float4(1.0, 1.0, 1.0, 1.0);
    float4 predicted_color = render_pixel_metal(x, y, agent_shape_params, agent_shape_colors, shapes_per_agent, background_color);
    uchar4 actual_color_uchar = target_image[y * image_width + x];
    float4 actual_color = float4(actual_color_uchar) / 255.0;
    float pixel_error = fabs(predicted_color.r - actual_color.r) +
                        fabs(predicted_color.g - actual_color.g) +
                        fabs(predicted_color.b - actual_color.b) +
                        fabs(predicted_color.a - actual_color.a);
    out_pixel_errors[thread_id] = pixel_error;
}

kernel void reduce_pixel_errors(
    const device float* pixel_errors, device float* out_agent_scores,
    const device uint* pixels_per_agent_ptr, uint agent_id [[thread_position_in_grid]]
){
    uint pixels_per_agent = *pixels_per_agent_ptr;
    uint start_index = agent_id * pixels_per_agent;
    float total_error = 0.0;
    for(uint i = 0; i < pixels_per_agent; ++i){
        total_error += pixel_errors[start_index + i];
    }
    out_agent_scores[agent_id] = total_error / pixels_per_agent;
}
"""