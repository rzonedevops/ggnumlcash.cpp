#include "market-data-memory.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using financial_market_data::HistoricalMarketDataFile;
using financial_market_data::MarketDataPool;
using financial_market_data::MarketDataRing;
using financial_market_data::Tick;

namespace {

void write_u32(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    for (unsigned int i = 0; i < 4; ++i) {
        bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
    }
}

void write_u64(std::vector<uint8_t> & bytes, size_t offset, uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) {
        bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
    }
}

void serialize_tick(std::vector<uint8_t> & bytes, size_t offset, const Tick & tick) {
    write_u64(bytes, offset, tick.timestamp_ns);
    write_u64(bytes, offset + 8, tick.instrument_id);
    write_u64(bytes, offset + 16, static_cast<uint64_t>(tick.bid_price_ticks));
    write_u64(bytes, offset + 24, static_cast<uint64_t>(tick.ask_price_ticks));
    write_u64(bytes, offset + 32, tick.bid_size);
    write_u64(bytes, offset + 40, tick.ask_size);
    write_u64(bytes, offset + 48, static_cast<uint64_t>(tick.last_price_ticks));
    write_u64(bytes, offset + 56, tick.last_size);
}

std::string temporary_path(const char * suffix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "market-data-memory-" + std::to_string(now) + suffix;
}

bool write_market_file(const std::string & path, const std::vector<Tick> & ticks) {
    std::vector<uint8_t> bytes(64 + ticks.size() * 64, 0);
    const char           magic[] = "GGMDATA1";
    for (size_t i = 0; i < sizeof(magic) - 1; ++i) {
        bytes[i] = static_cast<uint8_t>(magic[i]);
    }
    write_u32(bytes, 8, 1);
    write_u32(bytes, 12, 64);
    write_u64(bytes, 16, ticks.size());
    for (size_t i = 0; i < ticks.size(); ++i) {
        serialize_tick(bytes, 64 + i * 64, ticks[i]);
    }

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();
    return file.good();
}

bool test_pool() {
    MarketDataPool pool(2);
    Tick *         first  = pool.acquire();
    Tick *         second = pool.acquire();
    if (first == nullptr || second == nullptr || first == second ||
        reinterpret_cast<uintptr_t>(first) % alignof(Tick) != 0 || pool.acquire() != nullptr) {
        return false;
    }
    if (!pool.release(first) || pool.release(first) || pool.active_count() != 1 || pool.release(nullptr)) {
        return false;
    }

    Tick foreign;
    if (pool.release(&foreign)) {
        return false;
    }

    Tick *     reused = pool.acquire();
    const bool valid =
        reused == first && pool.reuse_count() == 1 && pool.allocation_count() == 3 && pool.active_count() == 2;
    return pool.release(second) && pool.release(reused) && valid && pool.active_count() == 0;
}

bool test_ring() {
    MarketDataRing ring(3);
    if (ring.push(Tick{ 1 }) == false || !ring.push(Tick{ 2 }) || !ring.push(Tick{ 3 }) || !ring.push(Tick{ 4 }) ||
        ring.size() != 3 || ring.at_oldest(3) != nullptr) {
        return false;
    }
    return ring.at_oldest(0)->timestamp_ns == 2 && ring.at_oldest(1)->timestamp_ns == 3 &&
           ring.at_oldest(2)->timestamp_ns == 4 && !MarketDataRing(0).push(Tick{});
}

bool test_historical_file() {
    const std::string       valid_path   = temporary_path(".bin");
    const std::string       invalid_path = temporary_path(".bad");
    const std::vector<Tick> expected     = {
        Tick{ 10, 2, -4, 6, 8, 10, -12, 14 },
        Tick{ 20, 3, -5, 7, 9, 11, -13, 15 },
    };
    if (!write_market_file(valid_path, expected)) {
        return false;
    }
    {
        std::vector<uint8_t> bytes(64, 0);
        const char           magic[] = "GGMDATA1";
        for (size_t i = 0; i < sizeof(magic) - 1; ++i) {
            bytes[i] = static_cast<uint8_t>(magic[i]);
        }
        write_u32(bytes, 8, 1);
        write_u32(bytes, 12, 64);
        write_u64(bytes, 16, 1);
        std::ofstream invalid(invalid_path, std::ios::binary);
        invalid.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    HistoricalMarketDataFile file;
    Tick                     tick;
    std::vector<Tick>        range;
    const bool valid = file.open(valid_path) && file.record_count() == expected.size() && file.read(1, tick) &&
                       tick.timestamp_ns == 20 && tick.bid_price_ticks == -5 && tick.last_size == 15 &&
                       !file.read(2, tick) && file.read_range(0, 2, range) && range.size() == 2 &&
                       range[0].instrument_id == 2 && !file.read_range(1, 2, range) &&
                       file.open(invalid_path) == false && !file.is_open();
    file.close();
    std::remove(valid_path.c_str());
    std::remove(invalid_path.c_str());
    return valid;
}

bool test_reuse_stability() {
    MarketDataPool pool(8);
    MarketDataRing ring(32);
    for (uint64_t i = 0; i < 100000; ++i) {
        Tick * tick = pool.acquire();
        if (tick == nullptr) {
            return false;
        }
        tick->timestamp_ns = i;
        if (!ring.push(*tick) || !pool.release(tick)) {
            return false;
        }
    }
    return pool.active_count() == 0 && pool.allocation_count() == 100000 && pool.reuse_count() == 99999 &&
           ring.at_oldest(ring.size() - 1)->timestamp_ns == 99999;
}

}  // namespace

int main() {
    if (!test_pool()) {
        std::cerr << "Market-data pool test failed" << std::endl;
        return 1;
    }
    if (!test_ring()) {
        std::cerr << "Market-data ring test failed" << std::endl;
        return 1;
    }
    if (!test_historical_file()) {
        std::cerr << "Historical file test failed" << std::endl;
        return 1;
    }
    if (!test_reuse_stability()) {
        std::cerr << "Market-data stability test failed" << std::endl;
        return 1;
    }
    std::cout << "Market-data memory tests passed" << std::endl;
    return 0;
}
