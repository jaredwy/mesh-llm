#pragma once
// Stand-in for <expected> so the fixture does not depend on the host
// standard library having it; the check matches by name.
namespace std {
template <typename T, typename E> class expected {};
template <typename E> class unexpected {};
} // namespace std
