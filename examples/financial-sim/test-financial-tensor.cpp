#include "financial-tensor.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

ggml_context * new_context() {
    ggml_init_params params = {};
    params.mem_size = 32 * 1024 * 1024;
    return ggml_init(params);
}

ggml_tensor * tensor(ggml_context * ctx, std::initializer_list<float> values) {
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, values.size());
    size_t i = 0;
    for (float value : values) {
        ggml_get_data_f32(result)[i++] = value;
    }
    return result;
}

void evaluate(ggml_context * ctx, ggml_tensor * result) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML graph computation failed");
    }
}

void expect_near(float actual, float expected, float tolerance, const std::string & name) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(name + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
    }
}

void test_black_scholes() {
    ggml_context * ctx = new_context();
    ggml_tensor * spot = tensor(ctx, {100.0f});
    ggml_tensor * strike = tensor(ctx, {100.0f});
    ggml_tensor * rate = tensor(ctx, {0.05f});
    ggml_tensor * dividend = tensor(ctx, {0.0f});
    ggml_tensor * volatility = tensor(ctx, {0.2f});
    ggml_tensor * years = tensor(ctx, {1.0f});

    ggml_tensor * call = ggnucash::financial::tensor::black_scholes_call(
        ctx, spot, strike, rate, dividend, volatility, years);
    ggml_tensor * put = ggnucash::financial::tensor::black_scholes_put(
        ctx, spot, strike, rate, dividend, volatility, years);
    ggml_tensor * delta = ggnucash::financial::tensor::black_scholes_call_delta(
        ctx, spot, strike, rate, dividend, volatility, years);
    ggml_tensor * gamma = ggnucash::financial::tensor::black_scholes_gamma(
        ctx, spot, strike, rate, dividend, volatility, years);
    ggml_tensor * vega = ggnucash::financial::tensor::black_scholes_vega(
        ctx, spot, strike, rate, dividend, volatility, years);

    evaluate(ctx, call);
    evaluate(ctx, put);
    evaluate(ctx, delta);
    evaluate(ctx, gamma);
    evaluate(ctx, vega);
    expect_near(ggml_get_data_f32(call)[0], 10.4506f, 0.002f, "Black-Scholes call");
    expect_near(ggml_get_data_f32(put)[0], 5.5735f, 0.002f, "Black-Scholes put");
    expect_near(ggml_get_data_f32(delta)[0], 0.6368f, 0.001f, "Black-Scholes call delta");
    expect_near(ggml_get_data_f32(gamma)[0], 0.01876f, 0.0001f, "Black-Scholes gamma");
    expect_near(ggml_get_data_f32(vega)[0], 37.524f, 0.02f, "Black-Scholes vega");
    ggml_free(ctx);
}

void test_risk_statistics_and_portfolio() {
    ggml_context * ctx = new_context();
    ggml_tensor * values = tensor(ctx, {1.0f, 2.0f, 3.0f, 4.0f});
    ggml_tensor * other = tensor(ctx, {2.0f, 4.0f, 6.0f, 8.0f});
    ggml_tensor * weights = tensor(ctx, {0.6f, 0.4f});
    ggml_tensor * expected_returns = tensor(ctx, {0.1f, 0.2f});
    ggml_tensor * covariance = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    float * covariance_data = ggml_get_data_f32(covariance);
    covariance_data[0] = 0.04f;
    covariance_data[1] = 0.0f;
    covariance_data[2] = 0.0f;
    covariance_data[3] = 0.09f;

    ggml_tensor * mean = ggnucash::financial::tensor::mean(ctx, values);
    ggml_tensor * variance = ggnucash::financial::tensor::population_variance(ctx, values);
    ggml_tensor * sample_variance = ggnucash::financial::tensor::sample_variance(ctx, values);
    ggml_tensor * correlation = ggnucash::financial::tensor::correlation(ctx, values, other);
    ggml_tensor * portfolio_return = ggnucash::financial::tensor::expected_portfolio_return(
        ctx, weights, expected_returns);
    ggml_tensor * portfolio_variance = ggnucash::financial::tensor::portfolio_variance(
        ctx, weights, covariance);

    evaluate(ctx, mean);
    evaluate(ctx, variance);
    evaluate(ctx, sample_variance);
    evaluate(ctx, correlation);
    evaluate(ctx, portfolio_return);
    evaluate(ctx, portfolio_variance);
    expect_near(ggml_get_data_f32(mean)[0], 2.5f, 1e-6f, "mean");
    expect_near(ggml_get_data_f32(variance)[0], 1.25f, 1e-6f, "population variance");
    expect_near(ggml_get_data_f32(sample_variance)[0], 1.6666667f, 1e-5f, "sample variance");
    expect_near(ggml_get_data_f32(correlation)[0], 1.0f, 1e-5f, "correlation");
    expect_near(ggml_get_data_f32(portfolio_return)[0], 0.14f, 1e-6f, "portfolio return");
    expect_near(ggml_get_data_f32(portfolio_variance)[0], 0.0288f, 1e-6f, "portfolio variance");
    ggml_free(ctx);
}

void test_monte_carlo_and_payoffs() {
    ggml_context * ctx = new_context();
    ggml_tensor * initial_price = tensor(ctx, {100.0f});
    ggml_tensor * rate = tensor(ctx, {0.05f});
    ggml_tensor * volatility = tensor(ctx, {0.2f});
    ggml_tensor * years = tensor(ctx, {1.0f});
    ggml_tensor * normals = tensor(ctx, {-1.0f, 0.0f, 1.0f});
    ggml_tensor * terminal_prices = ggnucash::financial::tensor::gbm_terminal_prices(
        ctx, initial_price, rate, volatility, years, normals);
    evaluate(ctx, terminal_prices);
    expect_near(ggml_get_data_f32(terminal_prices)[0], 84.3665f, 0.002f, "GBM lower path");
    expect_near(ggml_get_data_f32(terminal_prices)[1], 103.0455f, 0.002f, "GBM mean path");
    expect_near(ggml_get_data_f32(terminal_prices)[2], 125.86f, 0.003f, "GBM upper path");

    ggml_tensor * outcomes = tensor(ctx, {90.0f, 100.0f, 110.0f});
    ggml_tensor * strike = tensor(ctx, {100.0f});
    ggml_tensor * payoff = ggnucash::financial::tensor::monte_carlo_call_payoffs(ctx, outcomes, strike);
    ggml_tensor * price = ggnucash::financial::tensor::monte_carlo_call_price(
        ctx, outcomes, strike, rate, years);
    evaluate(ctx, payoff);
    evaluate(ctx, price);
    expect_near(ggml_get_data_f32(payoff)[0], 0.0f, 1e-6f, "call payoff below strike");
    expect_near(ggml_get_data_f32(payoff)[1], 0.0f, 1e-6f, "call payoff at strike");
    expect_near(ggml_get_data_f32(payoff)[2], 10.0f, 1e-6f, "call payoff above strike");
    expect_near(ggml_get_data_f32(price)[0], 3.1705f, 0.001f, "Monte Carlo call price");
    ggml_free(ctx);
}

void test_edge_cases() {
    ggml_context * ctx = new_context();
    ggml_tensor * singleton = tensor(ctx, {1.0f});
    bool rejected = false;
    try {
        (void) ggnucash::financial::tensor::sample_variance(ctx, singleton);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("sample_variance accepted a single observation");
    }
    ggml_free(ctx);
}

} // namespace

int main() {
    try {
        test_black_scholes();
        test_risk_statistics_and_portfolio();
        test_monte_carlo_and_payoffs();
        test_edge_cases();
        std::cout << "Financial tensor operation tests passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
