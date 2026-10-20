#include "market-data-memory.h"

#include <cstring>
#include <limits>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace financial_market_data {

namespace {

constexpr size_t   file_header_size = 64;
constexpr size_t   disk_tick_size   = 64;
constexpr uint32_t file_version     = 1;
constexpr char     file_magic[8]    = { 'G', 'G', 'M', 'D', 'A', 'T', 'A', '1' };

uint32_t load_u32_le(const uint8_t * bytes) noexcept {
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t load_u64_le(const uint8_t * bytes) noexcept {
    uint64_t value = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    }
    return value;
}

int64_t load_i64_le(const uint8_t * bytes) noexcept {
    const uint64_t value = load_u64_le(bytes);
    if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return static_cast<int64_t>(value);
    }
    return -1 - static_cast<int64_t>(~value);
}

bool valid_header(const uint8_t * bytes, size_t size, size_t & record_count) noexcept {
    if (size < file_header_size || std::memcmp(bytes, file_magic, sizeof(file_magic)) != 0 ||
        load_u32_le(bytes + 8) != file_version || load_u32_le(bytes + 12) != disk_tick_size) {
        return false;
    }

    const uint64_t count     = load_u64_le(bytes + 16);
    const size_t   available = size - file_header_size;
    if (count > available / disk_tick_size || count != available / disk_tick_size || available % disk_tick_size != 0) {
        return false;
    }

    record_count = static_cast<size_t>(count);
    return true;
}

}  // namespace

MarketDataPool::MarketDataPool(size_t capacity) :
    slots_(capacity),
    free_slots_(capacity),
    in_use_(capacity, 0),
    used_before_(capacity, 0),
    free_count_(capacity) {
    for (size_t i = 0; i < capacity; ++i) {
        free_slots_[i] = capacity - i - 1;
    }
}

Tick * MarketDataPool::acquire() noexcept {
    if (free_count_ == 0) {
        return nullptr;
    }

    const size_t index = free_slots_[--free_count_];
    in_use_[index]     = 1;
    if (used_before_[index]) {
        ++reuse_count_;
    } else {
        used_before_[index] = 1;
    }
    ++active_count_;
    ++allocation_count_;
    return &slots_[index];
}

bool MarketDataPool::release(Tick * tick) noexcept {
    if (tick == nullptr || slots_.empty()) {
        return false;
    }

    const uintptr_t start   = reinterpret_cast<uintptr_t>(slots_.data());
    const uintptr_t address = reinterpret_cast<uintptr_t>(tick);
    const size_t    bytes   = slots_.size() * sizeof(Tick);
    if (address < start || address - start >= bytes) {
        return false;
    }

    const size_t offset = static_cast<size_t>(address - start);
    if (offset % sizeof(Tick) != 0) {
        return false;
    }

    const size_t index = offset / sizeof(Tick);
    if (!in_use_[index]) {
        return false;
    }

    in_use_[index]             = 0;
    free_slots_[free_count_++] = index;
    --active_count_;
    return true;
}

MarketDataRing::MarketDataRing(size_t capacity) : ticks_(capacity) {}

bool MarketDataRing::push(const Tick & tick) noexcept {
    if (ticks_.empty()) {
        return false;
    }

    if (count_ < ticks_.size()) {
        ticks_[(oldest_ + count_) % ticks_.size()] = tick;
        ++count_;
    } else {
        ticks_[oldest_] = tick;
        oldest_         = (oldest_ + 1) % ticks_.size();
    }
    return true;
}

const Tick * MarketDataRing::at_oldest(size_t index) const noexcept {
    if (index >= count_) {
        return nullptr;
    }
    return &ticks_[(oldest_ + index) % ticks_.size()];
}

HistoricalMarketDataFile::~HistoricalMarketDataFile() {
    close();
}

bool HistoricalMarketDataFile::open(const std::string & path) {
    close();

#ifdef _WIN32
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < static_cast<LONGLONG>(file_header_size) ||
        static_cast<uint64_t>(file_size.QuadPart) > std::numeric_limits<size_t>::max()) {
        CloseHandle(file);
        return false;
    }

    const size_t size    = static_cast<size_t>(file_size.QuadPart);
    HANDLE       mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    if (mapping == nullptr) {
        return false;
    }

    const uint8_t * view = static_cast<const uint8_t *>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    CloseHandle(mapping);
    if (view == nullptr) {
        return false;
    }
#else
    const int file = ::open(path.c_str(), O_RDONLY);
    if (file < 0) {
        return false;
    }

    struct stat file_stat;
    if (fstat(file, &file_stat) != 0 || file_stat.st_size < static_cast<off_t>(file_header_size) ||
        static_cast<uint64_t>(file_stat.st_size) > std::numeric_limits<size_t>::max()) {
        ::close(file);
        return false;
    }

    const size_t size = static_cast<size_t>(file_stat.st_size);
    void *       view = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file, 0);
    ::close(file);
    if (view == MAP_FAILED) {
        return false;
    }
    const uint8_t * bytes = static_cast<const uint8_t *>(view);
#endif

#ifdef _WIN32
    const uint8_t * bytes = view;
#endif
    size_t records = 0;
    if (!valid_header(bytes, size, records)) {
#ifdef _WIN32
        UnmapViewOfFile(view);
#else
        munmap(view, size);
#endif
        return false;
    }

    mapping_      = bytes;
    mapped_size_  = size;
    record_count_ = records;
    return true;
}

void HistoricalMarketDataFile::close() noexcept {
    if (mapping_ == nullptr) {
        return;
    }
#ifdef _WIN32
    UnmapViewOfFile(mapping_);
#else
    munmap(const_cast<uint8_t *>(mapping_), mapped_size_);
#endif
    mapping_      = nullptr;
    mapped_size_  = 0;
    record_count_ = 0;
}

bool HistoricalMarketDataFile::read(size_t index, Tick & tick) const noexcept {
    if (mapping_ == nullptr || index >= record_count_) {
        return false;
    }

    const uint8_t * bytes = mapping_ + file_header_size + index * disk_tick_size;
    tick.timestamp_ns     = load_u64_le(bytes);
    tick.instrument_id    = load_u64_le(bytes + 8);
    tick.bid_price_ticks  = load_i64_le(bytes + 16);
    tick.ask_price_ticks  = load_i64_le(bytes + 24);
    tick.bid_size         = load_u64_le(bytes + 32);
    tick.ask_size         = load_u64_le(bytes + 40);
    tick.last_price_ticks = load_i64_le(bytes + 48);
    tick.last_size        = load_u64_le(bytes + 56);
    return true;
}

bool HistoricalMarketDataFile::read_range(size_t first, size_t count, std::vector<Tick> & ticks) const {
    if (mapping_ == nullptr || first > record_count_ || count > record_count_ - first) {
        return false;
    }

    std::vector<Tick> result(count);
    for (size_t i = 0; i < count; ++i) {
        if (!read(first + i, result[i])) {
            return false;
        }
    }
    ticks.swap(result);
    return true;
}

}  // namespace financial_market_data
