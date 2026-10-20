// T5 (docs/TYPE_DESIGN.md): the only road from native text to a const char*.
// ZString owns NUL-free bytes, so c_str() is total; CStringTable owns the
// strings and the pointer array a C ABI wants, and re-derives its pointers
// after a move (short strings live inside the std::string object).
#pragma once

#include <algorithm>
#include <expected>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace skippy {

enum class CStringError { EmbeddedNul };

// Owned bytes with no interior NUL: c_str() is total.
class ZString {
public:
    [[nodiscard]] static std::expected<ZString, CStringError> from(std::string_view bytes) {
        if (std::ranges::contains(bytes, '\0')) return std::unexpected(CStringError::EmbeddedNul);
        return ZString{std::string{bytes}};
    }
    [[nodiscard]] const char* c_str() const noexcept { return bytes_.c_str(); }
    [[nodiscard]] std::string_view view() const noexcept { return bytes_; }

private:
    explicit ZString(std::string bytes) noexcept : bytes_(std::move(bytes)) {}
    std::string bytes_;
};

// Owns the strings and the pointer array; pointers are valid while *this lives.
class CStringTable {
public:
    [[nodiscard]] static std::expected<CStringTable, CStringError>
    from(std::span<const std::string_view> values) {
        CStringTable table;
        table.owned_.reserve(values.size());
        // the one loop: a ranges::to over an expected-yielding transform
        // needs C++26 to stay a pipeline
        for (const std::string_view value : values) {
            auto z = ZString::from(value);
            if (!z) return std::unexpected(z.error());
            table.owned_.push_back(std::move(*z));
        }
        table.pointers_.reserve(table.owned_.size());
        std::ranges::transform(table.owned_, std::back_inserter(table.pointers_), &ZString::c_str);
        return table;
    }

    [[nodiscard]] std::span<const char* const> pointers() const noexcept { return pointers_; }
    [[nodiscard]] const char* const* data_or_null() const noexcept {
        return pointers_.empty() ? nullptr : pointers_.data();
    }
    [[nodiscard]] std::size_t size() const noexcept { return pointers_.size(); }

    CStringTable(const CStringTable&) = delete; // pointers would alias the other table
    CStringTable& operator=(const CStringTable&) = delete;
    CStringTable(CStringTable&& other) noexcept : owned_(std::move(other.owned_)) {
        derive_pointers();
    }
    CStringTable& operator=(CStringTable&& other) noexcept {
        if (this != &other) {
            owned_ = std::move(other.owned_);
            derive_pointers();
        }
        return *this;
    }

private:
    CStringTable() = default;
    void derive_pointers() {
        pointers_.clear();
        pointers_.reserve(owned_.size());
        std::ranges::transform(owned_, std::back_inserter(pointers_), &ZString::c_str);
    }
    std::vector<ZString> owned_;
    std::vector<const char*> pointers_;
};

} // namespace skippy
