/**
 * @file gguf_planner_probe.cpp
 * @brief Answers two questions about the native stage planner with one run:
 *        does it accept a plain GGUF as a single-shard package, and what does
 *        planning cost?
 *
 * Usage: skippy_api_planner_probe <model.gguf> [layer_count]
 *
 * The tensor inventory is read straight from the GGUF header; nothing here
 * uses llama to parse it, so a shape bug shows up as a planner rejection
 * rather than a misleading success.
 */

#include "llama/stage_planner.h"
#include "stage_load.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <expected>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct TensorInfo {
    std::string name;
    std::uint32_t type = 0;
    std::uint32_t dimension_count = 0;
    std::array<std::int64_t, 4> dimensions{};
    std::uint64_t offset = 0;
    std::uint64_t stored_length = 0;
};

class Reader {
public:
    explicit Reader(const std::string & path) : file_(path, std::ios::binary) {}

    [[nodiscard]] bool okay() const { return file_.good(); }
    [[nodiscard]] std::uint64_t tell() {
        return static_cast<std::uint64_t>(file_.tellg());
    }

    template <typename T>
    std::expected<void, std::string> read(T & value) {
        file_.read(reinterpret_cast<char *>(&value), sizeof(T));
        if (!file_) return std::unexpected("truncated header");
        return {};
    }

    std::expected<void, std::string> skip(std::uint64_t count) {
        file_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
        if (!file_) return std::unexpected("truncated header");
        return {};
    }

    std::expected<std::string, std::string> read_string() {
        std::uint64_t length = 0;
        if (auto ok = read(length); !ok) return std::unexpected(ok.error());
        if (length > (1U << 20)) return std::unexpected("implausible string length");
        std::string value(length, '\0');
        file_.read(value.data(), static_cast<std::streamsize>(length));
        if (!file_) return std::unexpected("truncated string");
        return value;
    }

private:
    std::ifstream file_;
};

// Skips one metadata value of the given GGUF type.
std::expected<void, std::string> skip_value(Reader & reader, std::uint32_t type) {
    switch (type) {
    case 0: return reader.skip(1);  // u8
    case 1: return reader.skip(1);  // i8
    case 2: return reader.skip(2);  // u16
    case 3: return reader.skip(2);  // i16
    case 4: return reader.skip(4);  // u32
    case 5: return reader.skip(4);  // i32
    case 6: return reader.skip(4);  // f32
    case 7: return reader.skip(1);  // bool
    case 10: return reader.skip(8); // u64
    case 11: return reader.skip(8); // i64
    case 12: return reader.skip(8); // f64
    case 8: {                       // string
        auto value = reader.read_string();
        if (!value) return std::unexpected(value.error());
        return {};
    }
    case 9: { // array
        std::uint32_t element_type = 0;
        std::uint64_t count = 0;
        if (auto ok = reader.read(element_type); !ok) return ok;
        if (auto ok = reader.read(count); !ok) return ok;
        for (std::uint64_t index = 0; index < count; ++index) {
            if (auto ok = skip_value(reader, element_type); !ok) return ok;
        }
        return {};
    }
    default: return std::unexpected("unknown metadata type " + std::to_string(type));
    }
}

std::expected<std::vector<TensorInfo>, std::string> read_inventory(const std::string & path) {
    Reader reader(path);
    if (!reader.okay()) return std::unexpected("cannot open " + path);

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (auto ok = reader.read(magic); !ok) return std::unexpected(ok.error());
    if (magic != 0x46554747U) return std::unexpected("not a GGUF file");
    if (auto ok = reader.read(version); !ok) return std::unexpected(ok.error());
    if (version < 2 || version > 3) return std::unexpected("unsupported GGUF version");
    if (auto ok = reader.read(tensor_count); !ok) return std::unexpected(ok.error());
    if (auto ok = reader.read(metadata_count); !ok) return std::unexpected(ok.error());

    std::uint64_t alignment = 32;
    for (std::uint64_t index = 0; index < metadata_count; ++index) {
        auto key = reader.read_string();
        if (!key) return std::unexpected(key.error());
        std::uint32_t type = 0;
        if (auto ok = reader.read(type); !ok) return std::unexpected(ok.error());
        if (*key == "general.alignment" && type == 4) {
            std::uint32_t value = 0;
            if (auto ok = reader.read(value); !ok) return std::unexpected(ok.error());
            alignment = value;
            continue;
        }
        if (auto ok = skip_value(reader, type); !ok) return std::unexpected(ok.error());
    }

    std::vector<TensorInfo> tensors;
    tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        auto name = reader.read_string();
        if (!name) return std::unexpected(name.error());
        TensorInfo info;
        info.name = std::move(*name);
        if (auto ok = reader.read(info.dimension_count); !ok) return std::unexpected(ok.error());
        if (info.dimension_count > 4) return std::unexpected("tensor rank above four");
        for (std::uint32_t dim = 0; dim < info.dimension_count; ++dim) {
            std::uint64_t extent = 0;
            if (auto ok = reader.read(extent); !ok) return std::unexpected(ok.error());
            info.dimensions[dim] = static_cast<std::int64_t>(extent);
        }
        if (auto ok = reader.read(info.type); !ok) return std::unexpected(ok.error());
        if (auto ok = reader.read(info.offset); !ok) return std::unexpected(ok.error());
        info.stored_length = ggml_row_size(static_cast<ggml_type>(info.type), info.dimensions[0]);
        for (std::uint32_t dim = 1; dim < info.dimension_count; ++dim) {
            info.stored_length *= static_cast<std::uint64_t>(info.dimensions[dim]);
        }
        if (info.name == "blk.0.attn_k.bias") {
            std::cout << "probe: " << info.name << " type=" << info.type << " dims="
                      << info.dimensions[0] << " offset=" << info.offset
                      << " stored=" << info.stored_length << "\n";
        }
        tensors.push_back(std::move(info));
    }
    // The native loader reports absolute file offsets: data section start
    // plus the tensor record's relative offset.
    const std::uint64_t data_start =
        (reader.tell() + alignment - 1) / alignment * alignment;
    for (TensorInfo & tensor : tensors) {
        tensor.offset += data_start;
    }
    // Package tensor ids must be strictly increasing.
    std::ranges::sort(tensors, {}, &TensorInfo::name);
    return tensors;
}

// Package identity is the content hash, so the probe hashes the shard.
std::string sha256_of(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    };
    std::array<std::uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto rotr = [](std::uint32_t value, int bits) {
        return (value >> bits) | (value << (32 - bits));
    };
    const std::uint64_t bit_length = static_cast<std::uint64_t>(data.size()) * 8U;
    data.push_back(0x80);
    while (data.size() % 64 != 56) data.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<std::uint8_t>(bit_length >> shift));
    }
    for (std::size_t block = 0; block < data.size(); block += 64) {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t index = 0; index < 16; ++index) {
            w[index] = (static_cast<std::uint32_t>(data[block + index * 4]) << 24) |
                       (static_cast<std::uint32_t>(data[block + index * 4 + 1]) << 16) |
                       (static_cast<std::uint32_t>(data[block + index * 4 + 2]) << 8) |
                       static_cast<std::uint32_t>(data[block + index * 4 + 3]);
        }
        for (std::size_t index = 16; index < 64; ++index) {
            const auto s0 = rotr(w[index - 15], 7) ^ rotr(w[index - 15], 18) ^ (w[index - 15] >> 3);
            const auto s1 = rotr(w[index - 2], 17) ^ rotr(w[index - 2], 19) ^ (w[index - 2] >> 10);
            w[index] = w[index - 16] + s0 + w[index - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, hh] = h;
        for (std::size_t index = 0; index < 64; ++index) {
            const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto ch = (e & f) ^ (~e & g);
            const auto t1 = hh + s1 + ch + k[index] + w[index];
            const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto maj = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string digest = "sha256:";
    for (const std::uint32_t word : h) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            digest.push_back(kHex[(word >> shift) & 0xfU]);
        }
    }
    return digest;
}

std::int64_t layer_count_from(const std::vector<TensorInfo> & tensors) {
    std::int64_t layers = 0;
    for (const TensorInfo & tensor : tensors) {
        const auto position = tensor.name.rfind("blk.");
        if (position == std::string::npos) continue;
        const auto dot = tensor.name.find('.', position);
        if (dot == std::string::npos) continue;
        layers = std::max(layers, std::stoll(tensor.name.substr(position + 4, dot - position - 4)) + 1);
    }
    return layers;
}

template <typename F>
double milliseconds(F && work) {
    const auto start = std::chrono::steady_clock::now();
    work();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double microseconds(std::size_t iterations, const std::function<void()> & work) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < iterations; ++index) work();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(stop - start).count() /
           static_cast<double>(iterations);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: skippy_api_planner_probe <model.gguf> [layer_count]\n";
        return 2;
    }
    const std::string path = argv[1];

    const auto inventory = read_inventory(path);
    if (!inventory) {
        std::cerr << "gguf: " << inventory.error() << "\n";
        return 2;
    }
    const std::int64_t layers =
        argc > 2 ? std::stoll(argv[2]) : layer_count_from(*inventory);
    std::cout << "gguf: " << inventory->size() << " tensors, " << layers << " layers\n";

    std::vector<skippy::PackageTensor> tensors;
    tensors.reserve(inventory->size());
    for (const TensorInfo & info : *inventory) {
        skippy::PackageTensor tensor;
        tensor.id = skippy::TensorId{info.name};
        tensor.native_name = info.name;
        tensor.ggml_type = static_cast<std::int32_t>(info.type);
        tensor.dimension_count = info.dimension_count;
        tensor.dimensions = info.dimensions;
        tensor.data_offset = info.offset;
        tensor.stored_length = info.stored_length;
        tensors.push_back(std::move(tensor));
    }

    skippy::StagePlannerInputs inputs;
    inputs.package = skippy::PackageId{sha256_of(path)};
    inputs.shard_paths = {path};
    inputs.tensors = std::move(tensors);
    // "decode" touches KV state, "embed" does not; asking for both shows the
    // per-family state annotations that patch 0004 added.
    inputs.profiles = {skippy::ProfileSpec{.id = skippy::ProfileId{"decode"},
                                           .tokens = 512,
                                           .sequences = 1,
                                           .outputs = 1},
                       skippy::ProfileSpec{.id = skippy::ProfileId{"embed"},
                                           .tokens = 512,
                                           .sequences = 1,
                                           .outputs = 512}};
    inputs.graph_configuration = skippy::GraphConfigurationId{"probe"};
    inputs.backend = skippy::BackendId{"metal"};

    skippy::llama::StagePlanner planner;
    double create_ms = 0.0;
    create_ms = milliseconds([&] {
        auto created = skippy::llama::StagePlanner::create(inputs);
        if (!created) {
            std::cerr << "planner rejected: " << created.error().message << "\n";
            return;
        }
        planner = std::move(*created);
    });
    if (!planner.valid()) {
        std::cerr << "create: " << create_ms << " ms (failed)\n";
        return 1;
    }
    std::cout << "create: " << create_ms << " ms\n";

    double realize_ms = 0.0;
    auto plan = std::expected<skippy::llama::StagePlanHandle, skippy::PlanFailure>{};
    realize_ms = milliseconds([&] {
        const auto start = argc > 3 ? std::stoi(argv[3]) : 0;
        const auto end = argc > 4 ? std::stoi(argv[4]) : static_cast<std::int32_t>(layers);
        plan = planner.realize(skippy::LayerRange{start, end});
    });
    if (!plan) {
        std::cerr << "realize rejected: " << plan.error().message << "\n";
        std::cerr << "realize: " << realize_ms << " ms (failed)\n";
        return 1;
    }
    std::cout << "realize: " << realize_ms << " ms\n";

    auto range = plan->range();
    if (!range) {
        std::cerr << "describe rejected: " << range.error().message << "\n";
        return 1;
    }
    std::cout << "range: [" << range->start << ", " << range->end << ")\n";

    const double materialize_ms = milliseconds([&] {
        if (const auto value = plan->materialize(); !value) {
            std::cerr << "materialize rejected: " << value.error().message << "\n";
        }
    });
    auto materialized = plan->materialize();
    if (!materialized) {
        std::cerr << "materialize rejected: " << materialized.error().message << "\n";
        return 1;
    }
    std::cout << "materialized: " << materialized->profiles.size() << " profiles, "
              << materialized->resident_tensors.size() << " resident tensors, "
              << materialized->execution_contract.size() << " contract bytes\n";

    for (const skippy::PlanProfile & profile : materialized->profiles) {
        const auto semantics = skippy::summarize(*materialized, profile.id);
        if (!semantics) continue;
        std::cout << "profile " << profile.id.value << ": " << profile.state_effects.size()
                  << " state effects, " << semantics->written << " written, layers ["
                  << semantics->first_layer << ", " << semantics->last_layer << "], recurrent="
                  << (semantics->recurrent ? "yes" : "no") << ", frontier="
                  << semantics->frontier_values << "\n";
    }

    // Gate for patch 0003: can the native loader open this stage using only
    // what the plan says (closure, contract, boundary identities)?
    if (argc > 4) {
        const auto load = skippy::stage_load_config(*materialized, materialized->profiles.front().id);
        if (!load) {
            std::cout << "staged open: config rejected (" << static_cast<int>(load.error()) << ")\n";
        } else {
            auto config = skippy_runtime_config_default();
            config.stage_index = load->stage_index;
            config.layer_start = load->layers.start;
            config.layer_end = load->layers.end;
            config.ctx_size = 512;
            config.n_batch = 512;
            config.n_ubatch = 512;
            config.n_gpu_layers = 999;
            config.load_mode = SKIPPY_LOAD_MODE_LAYER_PACKAGE;
            config.execution_contract = load->execution_contract.data();

            std::vector<const char *> residents;
            for (const std::string_view name : load->resident_tensors) {
                residents.push_back(name.data());
            }
            config.resident_tensor_names = residents.empty() ? nullptr : residents.data();
            config.resident_tensor_name_count = residents.size();

            std::vector<const char *> import_ids;
            std::vector<const char *> import_bindings;
            for (const std::string_view name : load->activation_import_identities) {
                import_ids.push_back(name.data());
            }
            for (const std::string_view name : load->activation_import_bindings) {
                import_bindings.push_back(name.data());
            }
            config.activation_import_identities = import_ids.empty() ? nullptr : import_ids.data();
            config.activation_import_identity_count = import_ids.size();
            config.activation_import_bindings = import_bindings.empty() ? nullptr : import_bindings.data();

            std::vector<const char *> export_ids;
            std::vector<const char *> export_bindings;
            for (const std::string_view name : load->activation_export_identities) {
                export_ids.push_back(name.data());
            }
            for (const std::string_view name : load->activation_export_bindings) {
                export_bindings.push_back(name.data());
            }
            config.activation_export_identities = export_ids.empty() ? nullptr : export_ids.data();
            config.activation_export_identity_count = export_ids.size();
            config.activation_export_bindings = export_bindings.empty() ? nullptr : export_bindings.data();

            skippy_model * model = nullptr;
            skippy_error * error = nullptr;
            const auto start_ms = std::chrono::steady_clock::now();
            const auto status = skippy_model_open(path.c_str(), &config, &model, &error);
            const double open_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - start_ms)
                                       .count();
            if (status != SKIPPY_STATUS_OK) {
                std::cout << "staged open: rejected after " << open_ms << " ms: "
                          << (error != nullptr && error->message != nullptr ? error->message : "?")
                          << "\n";
                if (error != nullptr) skippy_error_free(error);
            } else {
                std::cout << "staged open: ok in " << open_ms << " ms, layers ["
                          << config.layer_start << ", " << config.layer_end << ")\n";
                if (const auto freed = skippy_model_free(model, nullptr); freed != SKIPPY_STATUS_OK) {
                    std::cout << "staged free: status " << static_cast<int>(freed) << "\n";
                }
            }
        }
    }

    const double describe_us = microseconds(1000, [&] {
        const auto value = plan->range();
        if (!value) std::abort();
    });
    std::cout << "materialize: " << materialize_ms << " ms\n";
    std::cout << "describe: " << describe_us << " us/call\n";

    return 0;
}
