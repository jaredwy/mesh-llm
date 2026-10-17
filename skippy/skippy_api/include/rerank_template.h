#pragma once

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

namespace skippy::detail {

inline constexpr std::string_view rerank_query_marker = "{query}";
inline constexpr std::string_view rerank_document_marker = "{document}";

constexpr void append_replacing(std::string& out, std::string_view text, std::string_view marker,
                                std::string_view replacement) {
    bool first_piece = true;
    std::ranges::for_each(text | std::views::split(marker), [&](auto&& piece) {
        if (!std::exchange(first_piece, false)) {
            out.append(replacement);
        }
        out.append(std::string_view{piece.begin(), piece.end()});
    });
}

[[nodiscard]] constexpr std::string render_rerank_template(std::string_view prompt_template,
                                                           std::string_view query,
                                                           std::string_view document) {
    std::string rendered;

    rendered.reserve(prompt_template.size() + query.size() + document.size());

    bool first_piece = true;
    std::ranges::for_each(
        prompt_template | std::views::split(rerank_document_marker), [&](auto&& piece) {
            if (!std::exchange(first_piece, false)) {
                rendered.append(document);
            }
            append_replacing(rendered, std::string_view{piece.begin(), piece.end()},
                             rerank_query_marker, query);
        });
    return rendered;
}

} // namespace skippy::detail
