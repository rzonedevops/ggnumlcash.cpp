#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ggnucash {
namespace financial {
namespace tensor {

inline ggml_tensor * constant(ggml_context * ctx, float value) {
    ggml_tensor * result         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_get_data_f32(result)[0] = value;
    return result;
}

inline ggml_tensor * add(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (!ggml_can_repeat(b, a) && ggml_can_repeat(a, b)) {
        std::swap(a, b);
    }
    return ggml_add(ctx, a, b);
}

inline ggml_tensor * subtract(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (!ggml_can_repeat(b, a) && ggml_can_repeat(a, b)) {
        a = ggml_repeat(ctx, a, b);
    }
    return ggml_sub(ctx, a, b);
}

inline ggml_tensor * multiply(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (!ggml_can_repeat(b, a) && ggml_can_repeat(a, b)) {
        std::swap(a, b);
    }
    return ggml_mul(ctx, a, b);
}

inline ggml_tensor * divide(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (!ggml_can_repeat(b, a) && ggml_can_repeat(a, b)) {
        a = ggml_repeat(ctx, a, b);
    }
    return ggml_div(ctx, a, b);
}

inline ggml_tensor * scale(ggml_context * ctx, ggml_tensor * value, float factor) {
    return ggml_scale(ctx, value, factor);
}

inline ggml_tensor * sum(ggml_context * ctx, ggml_tensor * values) {
    return ggml_sum(ctx, values);
}

inline ggml_tensor * mean(ggml_context * ctx, ggml_tensor * values) {
    return ggml_mean(ctx, values);
}

inline ggml_tensor * square(ggml_context * ctx, ggml_tensor * value) {
    return ggml_sqr(ctx, value);
}

inline ggml_tensor * cube(ggml_context * ctx, ggml_tensor * value) {
    return multiply(ctx, square(ctx, value), value);
}

inline ggml_tensor * fourth_power(ggml_context * ctx, ggml_tensor * value) {
    return square(ctx, square(ctx, value));
}

inline ggml_tensor * positive_part(ggml_context * ctx, ggml_tensor * value) {
    return ggml_clamp(ctx, value, 0.0f, 3.402823466e+38f);
}

inline ggml_tensor * negative_part(ggml_context * ctx, ggml_tensor * value) {
    return ggml_clamp(ctx, value, -3.402823466e+38f, 0.0f);
}

inline ggml_tensor * discount_factor(ggml_context * ctx, ggml_tensor * rate, ggml_tensor * years) {
    ggml_tensor * one_plus_rate = add(ctx, constant(ctx, 1.0f), multiply(ctx, rate, years));
    return divide(ctx, constant(ctx, 1.0f), one_plus_rate);
}

inline ggml_tensor * continuous_discount_factor(ggml_context * ctx, ggml_tensor * rate, ggml_tensor * years) {
    return ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, rate, years)));
}

inline ggml_tensor * present_value(ggml_context * ctx,
                                   ggml_tensor *  future_cash_flow,
                                   ggml_tensor *  rate,
                                   ggml_tensor *  years) {
    return multiply(ctx, future_cash_flow, continuous_discount_factor(ctx, rate, years));
}

inline ggml_tensor * future_value(ggml_context * ctx,
                                  ggml_tensor *  present_cash_flow,
                                  ggml_tensor *  rate,
                                  ggml_tensor *  years) {
    return multiply(ctx, present_cash_flow, ggml_exp(ctx, multiply(ctx, rate, years)));
}

inline ggml_tensor * annuity_present_value(ggml_context * ctx,
                                           ggml_tensor *  payment,
                                           ggml_tensor *  rate,
                                           ggml_tensor *  periods) {
    ggml_tensor * one_plus_rate = add(ctx, constant(ctx, 1.0f), rate);
    ggml_tensor * remaining     = ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, periods, ggml_log(ctx, one_plus_rate))));
    return multiply(ctx, payment, divide(ctx, subtract(ctx, constant(ctx, 1.0f), remaining), rate));
}

inline ggml_tensor * annuity_future_value(ggml_context * ctx,
                                          ggml_tensor *  payment,
                                          ggml_tensor *  rate,
                                          ggml_tensor *  periods) {
    ggml_tensor * accumulation =
        ggml_exp(ctx, multiply(ctx, periods, ggml_log(ctx, add(ctx, constant(ctx, 1.0f), rate))));
    return multiply(ctx, payment, divide(ctx, subtract(ctx, accumulation, constant(ctx, 1.0f)), rate));
}

inline ggml_tensor * perpetuity_value(ggml_context * ctx, ggml_tensor * payment, ggml_tensor * rate) {
    return divide(ctx, payment, rate);
}

inline ggml_tensor * growing_perpetuity_value(ggml_context * ctx,
                                              ggml_tensor *  payment,
                                              ggml_tensor *  rate,
                                              ggml_tensor *  growth) {
    return divide(ctx, payment, subtract(ctx, rate, growth));
}

inline ggml_tensor * bond_price(ggml_context * ctx,
                                ggml_tensor *  face_value,
                                ggml_tensor *  coupon_rate,
                                ggml_tensor *  yield_rate,
                                ggml_tensor *  years) {
    ggml_tensor * coupon    = multiply(ctx, face_value, coupon_rate);
    ggml_tensor * coupons   = annuity_present_value(ctx, coupon, yield_rate, years);
    ggml_tensor * principal = multiply(
        ctx, face_value,
        ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, years, ggml_log(ctx, add(ctx, constant(ctx, 1.0f), yield_rate))))));
    return add(ctx, coupons, principal);
}

inline ggml_tensor * call_payoff(ggml_context * ctx, ggml_tensor * underlying, ggml_tensor * strike) {
    return positive_part(ctx, subtract(ctx, underlying, strike));
}

inline ggml_tensor * put_payoff(ggml_context * ctx, ggml_tensor * underlying, ggml_tensor * strike) {
    return positive_part(ctx, subtract(ctx, strike, underlying));
}

inline ggml_tensor * straddle_payoff(ggml_context * ctx, ggml_tensor * underlying, ggml_tensor * strike) {
    return add(ctx, call_payoff(ctx, underlying, strike), put_payoff(ctx, underlying, strike));
}

inline ggml_tensor * strangle_payoff(ggml_context * ctx,
                                     ggml_tensor *  underlying,
                                     ggml_tensor *  put_strike,
                                     ggml_tensor *  call_strike) {
    return add(ctx, put_payoff(ctx, underlying, put_strike), call_payoff(ctx, underlying, call_strike));
}

inline ggml_tensor * bull_call_spread_payoff(ggml_context * ctx,
                                             ggml_tensor *  underlying,
                                             ggml_tensor *  lower_strike,
                                             ggml_tensor *  upper_strike) {
    return subtract(ctx, call_payoff(ctx, underlying, lower_strike), call_payoff(ctx, underlying, upper_strike));
}

inline ggml_tensor * bear_call_spread_payoff(ggml_context * ctx,
                                             ggml_tensor *  underlying,
                                             ggml_tensor *  lower_strike,
                                             ggml_tensor *  upper_strike) {
    return subtract(ctx, call_payoff(ctx, underlying, upper_strike), call_payoff(ctx, underlying, lower_strike));
}

inline ggml_tensor * call_butterfly_payoff(ggml_context * ctx,
                                           ggml_tensor *  underlying,
                                           ggml_tensor *  lower_strike,
                                           ggml_tensor *  middle_strike,
                                           ggml_tensor *  upper_strike) {
    ggml_tensor * wings =
        add(ctx, call_payoff(ctx, underlying, lower_strike), call_payoff(ctx, underlying, upper_strike));
    return subtract(ctx, wings, scale(ctx, call_payoff(ctx, underlying, middle_strike), 2.0f));
}

inline ggml_tensor * put_butterfly_payoff(ggml_context * ctx,
                                          ggml_tensor *  underlying,
                                          ggml_tensor *  lower_strike,
                                          ggml_tensor *  middle_strike,
                                          ggml_tensor *  upper_strike) {
    ggml_tensor * wings =
        add(ctx, put_payoff(ctx, underlying, lower_strike), put_payoff(ctx, underlying, upper_strike));
    return subtract(ctx, wings, scale(ctx, put_payoff(ctx, underlying, middle_strike), 2.0f));
}

inline ggml_tensor * covered_call_payoff(ggml_context * ctx,
                                         ggml_tensor *  underlying,
                                         ggml_tensor *  entry_price,
                                         ggml_tensor *  strike) {
    return subtract(ctx, subtract(ctx, underlying, entry_price), call_payoff(ctx, underlying, strike));
}

inline ggml_tensor * protective_put_payoff(ggml_context * ctx,
                                           ggml_tensor *  underlying,
                                           ggml_tensor *  entry_price,
                                           ggml_tensor *  strike) {
    return add(ctx, subtract(ctx, underlying, entry_price), put_payoff(ctx, underlying, strike));
}

inline ggml_tensor * forward_price(ggml_context * ctx,
                                   ggml_tensor *  spot,
                                   ggml_tensor *  rate,
                                   ggml_tensor *  dividend_yield,
                                   ggml_tensor *  years) {
    return multiply(ctx, spot, ggml_exp(ctx, multiply(ctx, subtract(ctx, rate, dividend_yield), years)));
}

inline ggml_tensor * normal_pdf(ggml_context * ctx, ggml_tensor * value) {
    ggml_tensor * exponent = scale(ctx, square(ctx, value), -0.5f);
    return scale(ctx, ggml_exp(ctx, exponent), 0.3989422804014327f);
}

inline ggml_tensor * normal_cdf(ggml_context * ctx, ggml_tensor * value) {
    ggml_tensor * absolute = ggml_abs(ctx, value);
    ggml_tensor * t = divide(ctx, constant(ctx, 1.0f), add(ctx, constant(ctx, 1.0f), scale(ctx, absolute, 0.2316419f)));
    ggml_tensor * polynomial = add(ctx, scale(ctx, t, 1.330274429f), constant(ctx, -1.821255978f));
    polynomial               = add(ctx, multiply(ctx, polynomial, t), constant(ctx, 1.781477937f));
    polynomial               = add(ctx, multiply(ctx, polynomial, t), constant(ctx, -0.356563782f));
    polynomial               = add(ctx, multiply(ctx, polynomial, t), constant(ctx, 0.319381530f));
    ggml_tensor * tail       = multiply(ctx, normal_pdf(ctx, value), multiply(ctx, t, polynomial));
    ggml_tensor * signed_cdf = multiply(ctx, ggml_tanh(ctx, scale(ctx, value, 10000.0f)),
                                        subtract(ctx, constant(ctx, 1.0f), scale(ctx, tail, 2.0f)));
    return scale(ctx, add(ctx, constant(ctx, 1.0f), signed_cdf), 0.5f);
}

inline ggml_tensor * black_scholes_d1(ggml_context * ctx,
                                      ggml_tensor *  spot,
                                      ggml_tensor *  strike,
                                      ggml_tensor *  rate,
                                      ggml_tensor *  dividend_yield,
                                      ggml_tensor *  volatility,
                                      ggml_tensor *  years) {
    ggml_tensor * vol_sq_half = scale(ctx, square(ctx, volatility), 0.5f);
    ggml_tensor * drift       = add(ctx, subtract(ctx, rate, dividend_yield), vol_sq_half);
    ggml_tensor * numerator   = add(ctx, ggml_log(ctx, divide(ctx, spot, strike)), multiply(ctx, drift, years));
    ggml_tensor * denominator = multiply(ctx, volatility, ggml_sqrt(ctx, years));
    return divide(ctx, numerator, denominator);
}

inline ggml_tensor * black_scholes_d2(ggml_context * ctx,
                                      ggml_tensor *  spot,
                                      ggml_tensor *  strike,
                                      ggml_tensor *  rate,
                                      ggml_tensor *  dividend_yield,
                                      ggml_tensor *  volatility,
                                      ggml_tensor *  years) {
    return subtract(ctx, black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years),
                    multiply(ctx, volatility, ggml_sqrt(ctx, years)));
}

inline ggml_tensor * black_scholes_call(ggml_context * ctx,
                                        ggml_tensor *  spot,
                                        ggml_tensor *  strike,
                                        ggml_tensor *  rate,
                                        ggml_tensor *  dividend_yield,
                                        ggml_tensor *  volatility,
                                        ggml_tensor *  years) {
    ggml_tensor * d1 = black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years);
    ggml_tensor * d2 = subtract(ctx, d1, multiply(ctx, volatility, ggml_sqrt(ctx, years)));
    ggml_tensor * discounted_spot =
        multiply(ctx, spot, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years))));
    ggml_tensor * discounted_strike = multiply(ctx, strike, continuous_discount_factor(ctx, rate, years));
    return subtract(ctx, multiply(ctx, discounted_spot, normal_cdf(ctx, d1)),
                    multiply(ctx, discounted_strike, normal_cdf(ctx, d2)));
}

inline ggml_tensor * black_scholes_put(ggml_context * ctx,
                                       ggml_tensor *  spot,
                                       ggml_tensor *  strike,
                                       ggml_tensor *  rate,
                                       ggml_tensor *  dividend_yield,
                                       ggml_tensor *  volatility,
                                       ggml_tensor *  years) {
    ggml_tensor * d1 = black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years);
    ggml_tensor * d2 = subtract(ctx, d1, multiply(ctx, volatility, ggml_sqrt(ctx, years)));
    ggml_tensor * discounted_spot =
        multiply(ctx, spot, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years))));
    ggml_tensor * discounted_strike = multiply(ctx, strike, continuous_discount_factor(ctx, rate, years));
    return subtract(ctx, multiply(ctx, discounted_strike, normal_cdf(ctx, ggml_neg(ctx, d2))),
                    multiply(ctx, discounted_spot, normal_cdf(ctx, ggml_neg(ctx, d1))));
}

inline ggml_tensor * black_scholes_call_delta(ggml_context * ctx,
                                              ggml_tensor *  spot,
                                              ggml_tensor *  strike,
                                              ggml_tensor *  rate,
                                              ggml_tensor *  dividend_yield,
                                              ggml_tensor *  volatility,
                                              ggml_tensor *  years) {
    ggml_tensor * d1 = black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years);
    return multiply(ctx, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years))), normal_cdf(ctx, d1));
}

inline ggml_tensor * black_scholes_put_delta(ggml_context * ctx,
                                             ggml_tensor *  spot,
                                             ggml_tensor *  strike,
                                             ggml_tensor *  rate,
                                             ggml_tensor *  dividend_yield,
                                             ggml_tensor *  volatility,
                                             ggml_tensor *  years) {
    ggml_tensor * d1 = black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years);
    return multiply(ctx, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years))),
                    subtract(ctx, normal_cdf(ctx, d1), constant(ctx, 1.0f)));
}

inline ggml_tensor * black_scholes_gamma(ggml_context * ctx,
                                         ggml_tensor *  spot,
                                         ggml_tensor *  strike,
                                         ggml_tensor *  rate,
                                         ggml_tensor *  dividend_yield,
                                         ggml_tensor *  volatility,
                                         ggml_tensor *  years) {
    ggml_tensor * d1 = black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years);
    ggml_tensor * numerator =
        multiply(ctx, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years))), normal_pdf(ctx, d1));
    return divide(ctx, numerator, multiply(ctx, spot, multiply(ctx, volatility, ggml_sqrt(ctx, years))));
}

inline ggml_tensor * black_scholes_vega(ggml_context * ctx,
                                        ggml_tensor *  spot,
                                        ggml_tensor *  strike,
                                        ggml_tensor *  rate,
                                        ggml_tensor *  dividend_yield,
                                        ggml_tensor *  volatility,
                                        ggml_tensor *  years) {
    ggml_tensor * d1 = black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years);
    ggml_tensor * discounted_spot =
        multiply(ctx, spot, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years))));
    return multiply(ctx, multiply(ctx, discounted_spot, normal_pdf(ctx, d1)), ggml_sqrt(ctx, years));
}

inline ggml_tensor * black_scholes_call_theta(ggml_context * ctx,
                                              ggml_tensor *  spot,
                                              ggml_tensor *  strike,
                                              ggml_tensor *  rate,
                                              ggml_tensor *  dividend_yield,
                                              ggml_tensor *  volatility,
                                              ggml_tensor *  years) {
    ggml_tensor * d1    = black_scholes_d1(ctx, spot, strike, rate, dividend_yield, volatility, years);
    ggml_tensor * d2    = subtract(ctx, d1, multiply(ctx, volatility, ggml_sqrt(ctx, years)));
    ggml_tensor * first = scale(
        ctx,
        multiply(ctx,
                 multiply(ctx, multiply(ctx, spot, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years)))),
                          normal_pdf(ctx, d1)),
                 divide(ctx, volatility, ggml_sqrt(ctx, years))),
        -0.5f);
    ggml_tensor * second = multiply(ctx, multiply(ctx, rate, strike),
                                    multiply(ctx, continuous_discount_factor(ctx, rate, years), normal_cdf(ctx, d2)));
    ggml_tensor * third  = multiply(
        ctx, multiply(ctx, dividend_yield, spot),
        multiply(ctx, ggml_exp(ctx, ggml_neg(ctx, multiply(ctx, dividend_yield, years))), normal_cdf(ctx, d1)));
    return add(ctx, subtract(ctx, first, second), third);
}

inline ggml_tensor * black_scholes_call_rho(ggml_context * ctx,
                                            ggml_tensor *  spot,
                                            ggml_tensor *  strike,
                                            ggml_tensor *  rate,
                                            ggml_tensor *  dividend_yield,
                                            ggml_tensor *  volatility,
                                            ggml_tensor *  years) {
    ggml_tensor * d2 = black_scholes_d2(ctx, spot, strike, rate, dividend_yield, volatility, years);
    return multiply(ctx, multiply(ctx, multiply(ctx, strike, years), continuous_discount_factor(ctx, rate, years)),
                    normal_cdf(ctx, d2));
}

inline ggml_tensor * black_scholes_put_rho(ggml_context * ctx,
                                           ggml_tensor *  spot,
                                           ggml_tensor *  strike,
                                           ggml_tensor *  rate,
                                           ggml_tensor *  dividend_yield,
                                           ggml_tensor *  volatility,
                                           ggml_tensor *  years) {
    ggml_tensor * d2 = black_scholes_d2(ctx, spot, strike, rate, dividend_yield, volatility, years);
    return scale(
        ctx,
        multiply(ctx, multiply(ctx, multiply(ctx, strike, years), continuous_discount_factor(ctx, rate, years)),
                 normal_cdf(ctx, ggml_neg(ctx, d2))),
        -1.0f);
}

inline ggml_tensor * population_variance(ggml_context * ctx, ggml_tensor * values) {
    ggml_tensor * centered = subtract(ctx, values, mean(ctx, values));
    return mean(ctx, square(ctx, centered));
}

inline ggml_tensor * sample_variance(ggml_context * ctx, ggml_tensor * values) {
    const int64_t count = ggml_nelements(values);
    if (count < 2) {
        throw std::invalid_argument("sample_variance requires at least two values");
    }
    return scale(ctx, population_variance(ctx, values), static_cast<float>(count) / (count - 1));
}

inline ggml_tensor * standard_deviation(ggml_context * ctx, ggml_tensor * values) {
    return ggml_sqrt(ctx, population_variance(ctx, values));
}

inline ggml_tensor * sample_standard_deviation(ggml_context * ctx, ggml_tensor * values) {
    return ggml_sqrt(ctx, sample_variance(ctx, values));
}

inline ggml_tensor * semivariance(ggml_context * ctx, ggml_tensor * returns, ggml_tensor * target) {
    ggml_tensor * downside = positive_part(ctx, subtract(ctx, target, returns));
    return mean(ctx, square(ctx, downside));
}

inline ggml_tensor * downside_deviation(ggml_context * ctx, ggml_tensor * returns, ggml_tensor * target) {
    return ggml_sqrt(ctx, semivariance(ctx, returns, target));
}

inline ggml_tensor * standard_error(ggml_context * ctx, ggml_tensor * values) {
    return divide(ctx, standard_deviation(ctx, values),
                  ggml_sqrt(ctx, constant(ctx, static_cast<float>(ggml_nelements(values)))));
}

inline ggml_tensor * skewness(ggml_context * ctx, ggml_tensor * values) {
    ggml_tensor * centered  = subtract(ctx, values, mean(ctx, values));
    ggml_tensor * deviation = standard_deviation(ctx, values);
    return divide(ctx, mean(ctx, cube(ctx, centered)), cube(ctx, deviation));
}

inline ggml_tensor * excess_kurtosis(ggml_context * ctx, ggml_tensor * values) {
    ggml_tensor * centered  = subtract(ctx, values, mean(ctx, values));
    ggml_tensor * deviation = standard_deviation(ctx, values);
    return subtract(ctx, divide(ctx, mean(ctx, fourth_power(ctx, centered)), fourth_power(ctx, deviation)),
                    constant(ctx, 3.0f));
}

inline ggml_tensor * covariance(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    return mean(ctx, multiply(ctx, subtract(ctx, a, mean(ctx, a)), subtract(ctx, b, mean(ctx, b))));
}

inline ggml_tensor * correlation(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * denominator = multiply(ctx, standard_deviation(ctx, a), standard_deviation(ctx, b));
    return divide(ctx, covariance(ctx, a, b), denominator);
}

inline ggml_tensor * parametric_value_at_risk(ggml_context * ctx,
                                              ggml_tensor *  returns,
                                              ggml_tensor *  positive_tail_z_score) {
    return subtract(ctx, multiply(ctx, standard_deviation(ctx, returns), positive_tail_z_score), mean(ctx, returns));
}

inline ggml_tensor * parametric_expected_shortfall(ggml_context * ctx,
                                                   ggml_tensor *  returns,
                                                   ggml_tensor *  positive_tail_z_score) {
    ggml_tensor * tail_probability = subtract(ctx, constant(ctx, 1.0f), normal_cdf(ctx, positive_tail_z_score));
    ggml_tensor * tail_loss        = divide(
        ctx, multiply(ctx, standard_deviation(ctx, returns), normal_pdf(ctx, positive_tail_z_score)), tail_probability);
    return subtract(ctx, tail_loss, mean(ctx, returns));
}

inline ggml_tensor * expected_portfolio_return(ggml_context * ctx,
                                               ggml_tensor *  weights,
                                               ggml_tensor *  expected_returns) {
    return sum(ctx, multiply(ctx, weights, expected_returns));
}

inline ggml_tensor * portfolio_variance(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * covariance_matrix) {
    ggml_tensor * covariance_times_weights = ggml_mul_mat(ctx, covariance_matrix, weights);
    return sum(ctx, multiply(ctx, weights, covariance_times_weights));
}

inline ggml_tensor * portfolio_standard_deviation(ggml_context * ctx,
                                                  ggml_tensor *  weights,
                                                  ggml_tensor *  covariance_matrix) {
    return ggml_sqrt(ctx, portfolio_variance(ctx, weights, covariance_matrix));
}

inline ggml_tensor * sharpe_ratio(ggml_context * ctx,
                                  ggml_tensor *  portfolio_return,
                                  ggml_tensor *  risk_free_rate,
                                  ggml_tensor *  portfolio_deviation) {
    return divide(ctx, subtract(ctx, portfolio_return, risk_free_rate), portfolio_deviation);
}

inline ggml_tensor * sortino_ratio(ggml_context * ctx,
                                   ggml_tensor *  portfolio_return,
                                   ggml_tensor *  risk_free_rate,
                                   ggml_tensor *  downside_deviation_value) {
    return divide(ctx, subtract(ctx, portfolio_return, risk_free_rate), downside_deviation_value);
}

inline ggml_tensor * portfolio_beta(ggml_context * ctx, ggml_tensor * portfolio_returns, ggml_tensor * market_returns) {
    return divide(ctx, covariance(ctx, portfolio_returns, market_returns), population_variance(ctx, market_returns));
}

inline ggml_tensor * tracking_error(ggml_context * ctx,
                                    ggml_tensor *  portfolio_returns,
                                    ggml_tensor *  benchmark_returns) {
    return standard_deviation(ctx, subtract(ctx, portfolio_returns, benchmark_returns));
}

inline ggml_tensor * information_ratio(ggml_context * ctx,
                                       ggml_tensor *  portfolio_returns,
                                       ggml_tensor *  benchmark_returns) {
    ggml_tensor * active_returns = subtract(ctx, portfolio_returns, benchmark_returns);
    return divide(ctx, mean(ctx, active_returns), standard_deviation(ctx, active_returns));
}

inline ggml_tensor * concentration_index(ggml_context * ctx, ggml_tensor * weights) {
    return sum(ctx, square(ctx, weights));
}

inline ggml_tensor * gross_exposure(ggml_context * ctx, ggml_tensor * positions) {
    return sum(ctx, ggml_abs(ctx, positions));
}

inline ggml_tensor * net_exposure(ggml_context * ctx, ggml_tensor * positions) {
    return sum(ctx, positions);
}

inline ggml_tensor * active_return(ggml_context * ctx,
                                   ggml_tensor *  portfolio_returns,
                                   ggml_tensor *  benchmark_returns) {
    return subtract(ctx, mean(ctx, portfolio_returns), mean(ctx, benchmark_returns));
}

inline ggml_tensor * mean_variance_objective(ggml_context * ctx,
                                             ggml_tensor *  expected_return_value,
                                             ggml_tensor *  variance,
                                             ggml_tensor *  risk_aversion) {
    return subtract(ctx, expected_return_value, scale(ctx, multiply(ctx, risk_aversion, variance), 0.5f));
}

inline ggml_tensor * risk_adjusted_return(ggml_context * ctx, ggml_tensor * expected_return_value, ggml_tensor * risk) {
    return divide(ctx, expected_return_value, risk);
}

inline ggml_tensor * gbm_terminal_prices(ggml_context * ctx,
                                         ggml_tensor *  initial_price,
                                         ggml_tensor *  rate,
                                         ggml_tensor *  volatility,
                                         ggml_tensor *  years,
                                         ggml_tensor *  standard_normals) {
    ggml_tensor * drift     = multiply(ctx, subtract(ctx, rate, scale(ctx, square(ctx, volatility), 0.5f)), years);
    ggml_tensor * diffusion = multiply(ctx, multiply(ctx, volatility, ggml_sqrt(ctx, years)), standard_normals);
    return multiply(ctx, initial_price, ggml_exp(ctx, add(ctx, drift, diffusion)));
}

inline ggml_tensor * monte_carlo_call_payoffs(ggml_context * ctx, ggml_tensor * terminal_prices, ggml_tensor * strike) {
    return call_payoff(ctx, terminal_prices, strike);
}

inline ggml_tensor * monte_carlo_put_payoffs(ggml_context * ctx, ggml_tensor * terminal_prices, ggml_tensor * strike) {
    return put_payoff(ctx, terminal_prices, strike);
}

inline ggml_tensor * monte_carlo_call_price(ggml_context * ctx,
                                            ggml_tensor *  terminal_prices,
                                            ggml_tensor *  strike,
                                            ggml_tensor *  rate,
                                            ggml_tensor *  years) {
    return present_value(ctx, mean(ctx, monte_carlo_call_payoffs(ctx, terminal_prices, strike)), rate, years);
}

inline ggml_tensor * monte_carlo_put_price(ggml_context * ctx,
                                           ggml_tensor *  terminal_prices,
                                           ggml_tensor *  strike,
                                           ggml_tensor *  rate,
                                           ggml_tensor *  years) {
    return present_value(ctx, mean(ctx, monte_carlo_put_payoffs(ctx, terminal_prices, strike)), rate, years);
}

inline ggml_tensor * monte_carlo_standard_error(ggml_context * ctx, ggml_tensor * payoffs) {
    return standard_error(ctx, payoffs);
}

}  // namespace tensor
}  // namespace financial
}  // namespace ggnucash
