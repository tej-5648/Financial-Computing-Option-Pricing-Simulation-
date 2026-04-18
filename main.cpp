#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

struct OptionInput {
    double S0{};      // Spot price
    double K{};       // Strike
    double r{};       // Risk-free rate
    double sigma{};   // Volatility
    double T{};       // Time to maturity (years)
    std::uint64_t nPaths{};
};

double normalCdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double blackScholesCall(const OptionInput& in) {
    const double sqrtT = std::sqrt(in.T);
    const double d1 = (std::log(in.S0 / in.K) + (in.r + 0.5 * in.sigma * in.sigma) * in.T) / (in.sigma * sqrtT);
    const double d2 = d1 - in.sigma * sqrtT;
    return in.S0 * normalCdf(d1) - in.K * std::exp(-in.r * in.T) * normalCdf(d2);
}

struct MonteCarloResult {
    double price{};
    double stdErr{};
    double elapsedMs{};
};

// Antithetic variates + control variate (using discounted terminal stock price).
MonteCarloResult monteCarloCallParallel(const OptionInput& in, unsigned int nThreads) {
    const auto t0 = std::chrono::steady_clock::now();

    if (nThreads == 0) {
        nThreads = 1;
    }
    nThreads = std::min<unsigned int>(nThreads, static_cast<unsigned int>(in.nPaths));

    const std::uint64_t totalPairs = in.nPaths / 2;
    const std::uint64_t remainder = in.nPaths % 2;

    const double drift = (in.r - 0.5 * in.sigma * in.sigma) * in.T;
    const double volTerm = in.sigma * std::sqrt(in.T);
    const double disc = std::exp(-in.r * in.T);

    std::vector<double> partialSum(nThreads, 0.0);
    std::vector<double> partialSq(nThreads, 0.0);

    auto worker = [&](unsigned int tid, std::uint64_t startPair, std::uint64_t endPair) {
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(tid) + 1ULL) * 0xbf58476d1ce4e5b9ULL);
        std::normal_distribution<double> norm(0.0, 1.0);

        double sum = 0.0;
        double sq = 0.0;

        // Control variate coefficient fixed at 1.0 for simplicity.
        // X = discounted call payoff
        // Y = discounted terminal stock, E[Y] = S0
        constexpr double c = 1.0;

        for (std::uint64_t i = startPair; i < endPair; ++i) {
            const double z = norm(rng);

            const double ST1 = in.S0 * std::exp(drift + volTerm * z);
            const double payoff1 = disc * std::max(ST1 - in.K, 0.0);
            const double cv1 = payoff1 - c * (disc * ST1 - in.S0);

            const double ST2 = in.S0 * std::exp(drift - volTerm * z);
            const double payoff2 = disc * std::max(ST2 - in.K, 0.0);
            const double cv2 = payoff2 - c * (disc * ST2 - in.S0);

            const double sample = 0.5 * (cv1 + cv2);
            sum += sample;
            sq += sample * sample;
        }

        partialSum[tid] = sum;
        partialSq[tid] = sq;
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads);

    const std::uint64_t base = totalPairs / nThreads;
    const std::uint64_t extra = totalPairs % nThreads;

    std::uint64_t offset = 0;
    for (unsigned int t = 0; t < nThreads; ++t) {
        const std::uint64_t count = base + (t < extra ? 1ULL : 0ULL);
        threads.emplace_back(worker, t, offset, offset + count);
        offset += count;
    }
    for (auto& th : threads) {
        th.join();
    }

    double sum = std::accumulate(partialSum.begin(), partialSum.end(), 0.0);
    double sq = std::accumulate(partialSq.begin(), partialSq.end(), 0.0);

    std::uint64_t nSamples = totalPairs;

    // Handle odd path count with one extra draw.
    if (remainder) {
        std::mt19937_64 rng(0x12345678ULL);
        std::normal_distribution<double> norm(0.0, 1.0);
        const double z = norm(rng);
        const double ST = in.S0 * std::exp(drift + volTerm * z);
        const double payoff = disc * std::max(ST - in.K, 0.0);
        const double sample = payoff - (disc * ST - in.S0);
        sum += sample;
        sq += sample * sample;
        ++nSamples;
    }

    const double mean = sum / static_cast<double>(nSamples);
    const double var = std::max(0.0, sq / static_cast<double>(nSamples) - mean * mean);
    const double stderr = std::sqrt(var / static_cast<double>(nSamples));

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return {mean, stderr, elapsedMs};
}

int main() {
    const OptionInput in {
        .S0 = 100.0,
        .K = 100.0,
        .r = 0.05,
        .sigma = 0.2,
        .T = 1.0,
        .nPaths = 4'000'000
    };

    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());

    const double bs = blackScholesCall(in);
    const MonteCarloResult mc1 = monteCarloCallParallel(in, 1);
    const MonteCarloResult mcp = monteCarloCallParallel(in, hw);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Financial Computing: Option Pricing (Monte Carlo) ===\n";
    std::cout << "S0=" << in.S0 << ", K=" << in.K << ", r=" << in.r
              << ", sigma=" << in.sigma << ", T=" << in.T
              << ", paths=" << in.nPaths << "\n\n";

    std::cout << "Black-Scholes call price : " << bs << "\n";
    std::cout << "MC (1 thread)           : " << mc1.price
              << "  [SE=" << mc1.stdErr << ", time=" << mc1.elapsedMs << " ms]"
              << "\n";
    std::cout << "MC (" << hw << " threads)       : " << mcp.price
              << "  [SE=" << mcp.stdErr << ", time=" << mcp.elapsedMs << " ms]"
              << "\n";

    if (mcp.elapsedMs > 0.0) {
        std::cout << "Speedup                 : " << (mc1.elapsedMs / mcp.elapsedMs) << "x\n";
    }
    std::cout << "Abs error vs B-S        : " << std::fabs(mcp.price - bs) << "\n";

    return 0;
}
