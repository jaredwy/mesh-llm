// T1 (docs/TYPE_DESIGN.md): a failure-only result with no polarity.
// expected<void, E> and optional<E> both convert to bool, and !result means
// the opposite in the two spellings; Fallible has no operator bool at all,
// so the flip does not compile.
#pragma once

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

namespace skippy {

template <typename E> class [[nodiscard]] Fallible {
public:
    static constexpr Fallible ok() noexcept { return Fallible{}; }
    static constexpr Fallible fail(E error) noexcept(std::is_nothrow_move_constructible_v<E>) {
        return Fallible{std::move(error)};
    }

    [[nodiscard]] constexpr bool failed() const noexcept { return error_.has_value(); }
    [[nodiscard]] constexpr const E& error() const& noexcept { return *error_; }
    [[nodiscard]] constexpr E&& error() && noexcept { return std::move(*error_); }

    template <std::invocable<const E&> F>
    [[nodiscard]] constexpr auto map_error(F&& f) const
        -> Fallible<std::invoke_result_t<F, const E&>> {
        using R = Fallible<std::invoke_result_t<F, const E&>>;
        return failed() ? R::fail(std::invoke(std::forward<F>(f), *error_)) : R::ok();
    }

private:
    constexpr Fallible() = default;
    constexpr explicit Fallible(E error) : error_(std::move(error)) {}
    std::optional<E> error_;
};

} // namespace skippy
