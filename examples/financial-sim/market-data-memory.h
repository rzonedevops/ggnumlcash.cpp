#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace financial_market_data {

struct alignas(64) Tick {
    uint64_t timestamp_ns     = 0;
    uint64_t instrument_id    = 0;
    int64_t  bid_price_ticks  = 0;
    int64_t  ask_price_ticks  = 0;
    uint64_t bid_size         = 0;
    uint64_t ask_size         = 0;
    int64_t  last_price_ticks = 0;
    uint64_t last_size        = 0;
};

static_assert(sizeof(Tick) == 64, "Tick must occupy one cache line");
static_assert(std::is_trivially_copyable<Tick>::value, "Tick must be trivially copyable");

class MarketDataPool {
  public:
    explicit MarketDataPool(size_t capacity);

    Tick * acquire() noexcept;
    bool   release(Tick * tick) noexcept;

    size_t capacity() const noexcept { return slots_.size(); }

    size_t active_count() const noexcept { return active_count_; }

    size_t allocation_count() const noexcept { return allocation_count_; }

    size_t reuse_count() const noexcept { return reuse_count_; }

  private:
    std::vector<Tick>    slots_;
    std::vector<size_t>  free_slots_;
    std::vector<uint8_t> in_use_;
    std::vector<uint8_t> used_before_;
    size_t               free_count_       = 0;
    size_t               active_count_     = 0;
    size_t               allocation_count_ = 0;
    size_t               reuse_count_      = 0;
};

class MarketDataRing {
  public:
    explicit MarketDataRing(size_t capacity);

    bool         push(const Tick & tick) noexcept;
    const Tick * at_oldest(size_t index) const noexcept;

    size_t capacity() const noexcept { return ticks_.size(); }

    size_t size() const noexcept { return count_; }

  private:
    std::vector<Tick> ticks_;
    size_t            oldest_ = 0;
    size_t            count_  = 0;
};

class HistoricalMarketDataFile {
  public:
    HistoricalMarketDataFile() = default;
    ~HistoricalMarketDataFile();

    HistoricalMarketDataFile(const HistoricalMarketDataFile &)             = delete;
    HistoricalMarketDataFile & operator=(const HistoricalMarketDataFile &) = delete;

    bool open(const std::string & path);
    void close() noexcept;
    bool read(size_t index, Tick & tick) const noexcept;
    bool read_range(size_t first, size_t count, std::vector<Tick> & ticks) const;

    bool is_open() const noexcept { return mapping_ != nullptr; }

    size_t record_count() const noexcept { return record_count_; }

  private:
    const uint8_t * mapping_      = nullptr;
    size_t          mapped_size_  = 0;
    size_t          record_count_ = 0;
};

}  // namespace financial_market_data
