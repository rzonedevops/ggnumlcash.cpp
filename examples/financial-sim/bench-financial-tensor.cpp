#include "financial-tensor.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

double normal_cdf(double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double black_scholes_call(double spot, double strike, double rate, double volatility, double years) {
    const double root_time = std::sqrt(years);
    const double d1 =
        (std::log(spot / strike) + (rate + volatility * volatility / 2.0) * years) / (volatility * root_time);
    const double d2 = d1 - volatility * root_time;
    return spot * normal_cdf(d1) - strike * std::exp(-rate * years) * normal_cdf(d2);
}

ggml_context * new_context(size_t bytes) {
    ggml_init_params params = {};
    params.mem_size         = bytes;
    return ggml_init(params);
}

}  // namespace

int main() {
    constexpr int64_t  count            = 100000;
    constexpr float    rate_value       = 0.05f;
    constexpr float    volatility_value = 0.2f;
    constexpr float    years_value      = 1.0f;
    constexpr float    strike_value     = 100.0f;
    std::vector<float> spots(count, 100.0f);

    const auto scalar_start = std::chrono::steady_clock::now();
    double     scalar_total = 0.0;
    for (float spot : spots) {
        scalar_total += black_scholes_call(spot, strike_value, rate_value, volatility_value, years_value);
    }
    const auto scalar_end = std::chrono::steady_clock::now();

    ggml_context * ctx         = new_context(32 * 1024 * 1024);
    ggml_tensor *  spot_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    for (int64_t i = 0; i < count; ++i) {
        ggml_get_data_f32(spot_tensor)[i] = spots[i];
    }
    ggml_tensor * strike     = ggnucash::financial::tensor::constant(ctx, strike_value);
    ggml_tensor * rate       = ggnucash::financial::tensor::constant(ctx, rate_value);
    ggml_tensor * dividend   = ggnucash::financial::tensor::constant(ctx, 0.0f);
    ggml_tensor * volatility = ggnucash::financial::tensor::constant(ctx, volatility_value);
    ggml_tensor * years      = ggnucash::financial::tensor::constant(ctx, years_value);
    ggml_tensor * result =
        ggnucash::financial::tensor::black_scholes_call(ctx, spot_tensor, strike, rate, dividend, volatility, years);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    const auto ggml_start = std::chrono::steady_clock::now();
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) {
        std::cerr << "GGML graph computation failed\n";
        ggml_free(ctx);
        return EXIT_FAILURE;
    }
    const auto ggml_end = std::chrono::steady_clock::now();

    const auto scalar_ms  = std::chrono::duration<double, std::milli>(scalar_end - scalar_start).count();
    const auto ggml_ms    = std::chrono::duration<double, std::milli>(ggml_end - ggml_start).count();
    double     ggml_total = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        ggml_total += ggml_get_data_f32(result)[i];
    }
    std::cout << "Black-Scholes call pricing for " << count << " contracts\n"
              << "  scalar C++ reference: " << scalar_ms << " ms (sum " << scalar_total << ")\n"
              << "  GGML CPU tensor graph: " << ggml_ms << " ms (sum " << ggml_total << ")\n"
              << "  absolute sum difference: " << std::fabs(ggml_total - scalar_total) << '\n';
    ggml_free(ctx);
    return 0;
}
