// SPDX-FileCopyrightText: Copyright 2025 KNACK Render Diagnostics
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/knack_render_diag.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "common/logging/log.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/texture_cache.h"

namespace KnackDiag {

// ─── Global state ──────────────────────────────────────────────────
Flags g_flags;
std::atomic<u64> g_frame_id{0};
std::atomic<u64> g_submit_id{0};
std::atomic<u64> g_draw_id{0};
std::atomic<u32> g_texture_dump_count{0};
std::atomic<bool> g_current_frame_is_effect{false};

static std::once_flag init_once;

void Initialize() {
    std::call_once(init_once, [] {
        g_flags = Flags::LoadFromEnv();
        if (g_flags.render_diag) {
            LOG_INFO(Common, "=== KNACK RENDER DIAGNOSTICS ENABLED ===");
            LOG_INFO(Common, "  effect_diag      = {}", g_flags.effect_diag);
            LOG_INFO(Common, "  texture_dump     = {}", g_flags.texture_dump);
            LOG_INFO(Common, "  fs_summary       = {}", g_flags.fs_summary);
            LOG_INFO(Common, "  pm4_trace_effects = {}", g_flags.pm4_trace_effects);
            LOG_INFO(Common, "  renderdoc_labels  = {}", g_flags.renderdoc_labels);
            LOG_INFO(Common, "=========================================");
            // Ensure dumps directory exists
            std::filesystem::create_directories("dumps/knack_effects");
            // Register FS summary write on exit
            if (g_flags.fs_summary) {
                std::atexit([] { KnackFsWriteSummary(); });
            }
        }
    });
}

bool IsEnabled() {
    Initialize();
    return g_flags.render_diag;
}

const Flags& GetFlags() {
    Initialize();
    return g_flags;
}

// ─── Effect type names ─────────────────────────────────────────────
const char* EffectTypeName(EffectType t) {
    switch (t) {
    case EffectType::ParticleTrail:    return "PARTICLE_TRAIL";
    case EffectType::Explosion:         return "EXPLOSION";
    case EffectType::SmokeCloud:       return "SMOKE_CLOUD";
    case EffectType::Sunstone:         return "SUNSTONE";
    case EffectType::TransparentEffect: return "TRANSPARENT_EFFECT";
    default:                            return "UNKNOWN";
    }
}

// ─── EffectDrawInfo ────────────────────────────────────────────────
EffectDrawInfo EffectDrawInfo::FromRegs(const AmdGpu::Regs& regs, u64 submit, u64 draw,
                                         size_t pkt_off, u32 pkt_idx) {
    EffectDrawInfo info{};
    info.submit_id = submit;
    info.draw_id = draw;
    info.packet_offset = pkt_off;
    info.packet_index = pkt_idx;
    info.effect_type = EffectType::Unknown;

    // Shader hashes from program addresses (proxy hashes until actual pipeline access)
    info.vs_hash = regs.vs_program.address;
    info.fs_hash = regs.ps_program.address;
    info.gs_hash = regs.gs_program.address;
    info.cs_hash = 0;

    // Blend state for MRT 0
    if constexpr (AmdGpu::NUM_COLOR_BUFFERS > 0) {
        const auto& bc = regs.blend_control[0];
        info.blend_enabled = bc.enable != 0;
        info.color_src_factor = static_cast<u32>(bc.color_src_factor);
        info.color_func = static_cast<u32>(bc.color_func);
        info.color_dst_factor = static_cast<u32>(bc.color_dst_factor);
        info.alpha_src_factor = static_cast<u32>(bc.alpha_src_factor);
        info.alpha_func = static_cast<u32>(bc.alpha_func);
        info.alpha_dst_factor = static_cast<u32>(bc.alpha_dst_factor);
    }

    // Write masks
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        info.write_mask[i] = regs.color_target_mask.GetMask(i);
    }

    // Use color_control mode as MRT proxy
    info.mrt_mask = static_cast<u32>(regs.color_control.mode);

    // Depth state
    info.depth_test = regs.depth_control.depth_enable;
    info.depth_write = regs.depth_control.depth_write_enable;

    // Draw params
    info.num_indices = regs.num_indices;
    info.num_instances = regs.num_instances.NumInstances();

    // Geometry shader presence
    info.has_geometry_shader = regs.stage_enable.gs_en != 0;

    // Primitive type
    info.prim_type = static_cast<u32>(regs.primitive_type);

    info.effect_type = info.Classify();
    return info;
}

bool EffectDrawInfo::IsEffect() const {
    return effect_type != EffectType::Unknown;
}

EffectType EffectDrawInfo::Classify() const {
    // Heuristic 1: Additive/transparent blending
    if (blend_enabled) {
        const bool is_additive =
            (color_src_factor == static_cast<u32>(AmdGpu::BlendControl::BlendFactor::SrcAlpha)) &&
            (color_dst_factor == static_cast<u32>(AmdGpu::BlendControl::BlendFactor::One));
        const bool is_alpha_blend =
            (color_src_factor == static_cast<u32>(AmdGpu::BlendControl::BlendFactor::SrcAlpha)) &&
            (color_dst_factor == static_cast<u32>(AmdGpu::BlendControl::BlendFactor::OneMinusSrcAlpha));

        // Additive blending → likely particle/explosion/sunstone
        if (is_additive) {
            // Sunstone often has larger draw calls, particles are many small draws
            if (num_indices < 20 && num_instances == 1) {
                return EffectType::ParticleTrail;
            }
            if (num_indices > 100) {
                return EffectType::Sunstone;
            }
            return EffectType::ParticleTrail;
        }

        // Alpha blend → could be transparent effect or smoke
        if (is_alpha_blend) {
            if (has_geometry_shader) {
                return EffectType::ParticleTrail;
            }
            return EffectType::TransparentEffect;
        }
    }

    // Heuristic 2: Has geometry shader (often used for particle effects in GNM)
    if (has_geometry_shader) {
        return EffectType::ParticleTrail;
    }

    // Heuristic 3: Small draw calls (billboards)
    if (num_indices <= 6 && num_instances == 1) {
        return EffectType::ParticleTrail;
    }

    return EffectType::Unknown;
}

void EffectDrawInfo::Log() const {
    LOG_DEBUG(Lib_GnmDriver,
              "KNACK_EFFECT_DRAW submit={} draw={} type={} pkt_off={} pkt_idx={} "
              "vs_hash=0x{:016x} fs_hash=0x{:016x} gs_hash=0x{:016x} "
              "blend={} src_factor={} dst_factor={} func={} "
              "alpha_src={} alpha_dst={} alpha_func={} "
              "write_mask=[{},{},{},{},{},{},{},{}] mrt_mask={} "
              "depth_test={} depth_write={} has_gs={} prim_type={} "
              "num_idx={} num_inst={}",
              submit_id, draw_id, EffectTypeName(effect_type),
              packet_offset, packet_index,
              vs_hash, fs_hash, gs_hash,
              blend_enabled, color_src_factor, color_dst_factor, color_func,
              alpha_src_factor, alpha_dst_factor, alpha_func,
              write_mask[0], write_mask[1], write_mask[2], write_mask[3],
              write_mask[4], write_mask[5], write_mask[6], write_mask[7],
              mrt_mask,
              depth_test, depth_write, has_geometry_shader, prim_type,
              num_indices, num_instances);
}

// ─── Texture descriptor logging ────────────────────────────────────
void TextureDiagInfo::Log(const char* prefix) const {
    LOG_DEBUG(Lib_GnmDriver,
              "{}_TEX id={} gpu_addr=0x{:016x} {}x{}x{} fmt={} tile={} mips={} "
              "samples={} pitch={} usage=0x{:x} srgb={} depth={}",
              prefix, image_id, gpu_address, width, height, depth,
              pixel_format, tile_mode, mip_count, num_samples, pitch,
              usage_flags, is_srgb, is_depth);
}

void LogBoundTexture(u32 slot, const VideoCore::Image& image, const char* prefix) {
    TextureDiagInfo t{};
    t.image_id = static_cast<u32>(image.image_uid); // approximate
    t.gpu_address = image.info.guest_address;
    t.width = image.info.size.width;
    t.height = image.info.size.height;
    t.depth = image.info.size.depth;
    t.pitch = image.info.pitch;
    t.pixel_format = static_cast<u32>(image.info.pixel_format);
    t.tile_mode = static_cast<u32>(image.info.tile_mode);
    t.array_mode = static_cast<u32>(image.info.array_mode);
    t.mip_count = image.info.resources.num_mips;
    t.num_samples = image.info.num_samples;
    t.is_srgb = false; // TODO: detect from format
    t.is_tiled = image.info.props.is_tiled;
    t.is_depth = image.info.props.is_depth;
    t.usage_flags = static_cast<u32>(image.usage_flags);

    LOG_DEBUG(Lib_GnmDriver,
              "{} slot={} gpu_addr=0x{:016x} size={}x{}x{} fmt={} tile={} "
              "mips={} samples={} pitch={} usage=0x{:x} is_depth={}",
              prefix, slot, t.gpu_address, t.width, t.height, t.depth,
              t.pixel_format, t.tile_mode, t.mip_count, t.num_samples,
              t.pitch, t.usage_flags, t.is_depth);
}

void LogAllBoundTextures(u32 first_image_idx, u32 image_count, VideoCore::TextureCache& cache) {
    // This is a simplified version — full implementation would iterate
    // through all bound images in the rasterizer
    LOG_DEBUG(Lib_GnmDriver, "KNACK_BOUND_TEXTURES first_idx={} count={}",
              first_image_idx, image_count);
}

// ─── PM4 drift trace ───────────────────────────────────────────────
void LogPm4DriftContext(std::span<const u32> dcb, size_t current_offset,
                         u32 header_raw, u32 type, bool skipped) {
    if (!g_flags.pm4_trace_effects) return;

    const u32* data = dcb.data();
    const size_t remaining = dcb.size();

    LOG_DEBUG(Lib_GnmDriver,
              "KNACK_PM4_DRIFT off={} rem={} header=0x{:08x} type={} skipped={}",
              current_offset, remaining, header_raw, type, skipped);

    // Previous 8 dwords
    if (current_offset >= 1) {
        const size_t start = (current_offset >= 8) ? (current_offset - 8) : 0;
        const size_t n = std::min<size_t>(current_offset - start, 8);
        for (size_t i = 0; i < n; ++i) {
            LOG_DEBUG(Lib_GnmDriver, "KNACK_PM4_DRIFT_PREV[{}] = 0x{:08x}", i, data[start + i]);
        }
    }

    // Is this inside a NOP payload?
    {
        bool is_nop = false;
        // Look backward for a Nop header: count=57, opcode=0x10
        if (current_offset >= 1 && data[current_offset - 1] == 0xc0391000) {
            is_nop = true;
        }
        if (current_offset >= 2 && data[current_offset - 2] == 0xc0391000) {
            is_nop = true;
        }
        if (is_nop) {
            LOG_DEBUG(Lib_GnmDriver, "KNACK_PM4_DRIFT_INSIDE_NOP_PAYLOAD offset={}", current_offset);
        }
    }

    // Next 8 dwords (or less)
    {
        const size_t max_dump = std::min<size_t>(remaining, 8);
        for (size_t i = 0; i < max_dump; ++i) {
            LOG_DEBUG(Lib_GnmDriver, "KNACK_PM4_DRIFT_NEXT[{}] = 0x{:08x}", i, data[current_offset + i]);
        }
    }
}

// ─── Texture dumping ───────────────────────────────────────────────
void DumpEffectTextures(const EffectDrawInfo& info, VideoCore::TextureCache& cache,
                         const std::vector<std::pair<u32, VideoCore::ImageId>>& images) {
    if (!g_flags.texture_dump || !g_flags.effect_diag) return;

    const u32 count = g_texture_dump_count.fetch_add(1);
    if (count >= MAX_TEXTURE_DUMPS) {
        if (count == MAX_TEXTURE_DUMPS) {
            LOG_DEBUG(Lib_GnmDriver, "KNACK_TEXTURE_DUMP_LIMIT reached: {} dumps", MAX_TEXTURE_DUMPS);
        }
        return;
    }

    // Create dump directory
    std::string dir = fmt::format("dumps/knack_effects/frame{:04d}_draw{:04d}_type{}",
                                   g_frame_id.load(), info.draw_id,
                                   EffectTypeName(info.effect_type));
    std::filesystem::create_directories(dir);

    // Write descriptor JSON
    std::string json_path = dir + "/descriptors.json";
    std::ofstream json(json_path);
    json << "{" << std::endl;
    json << "  \"submit_id\": " << info.submit_id << "," << std::endl;
    json << "  \"draw_id\": " << info.draw_id << "," << std::endl;
    json << "  \"effect_type\": \"" << EffectTypeName(info.effect_type) << "\"," << std::endl;
    json << "  \"vs_hash\": \"0x" << fmt::format("{:016x}", info.vs_hash) << "\"," << std::endl;
    json << "  \"fs_hash\": \"0x" << fmt::format("{:016x}", info.fs_hash) << "\"," << std::endl;
    json << "  \"gs_hash\": \"0x" << fmt::format("{:016x}", info.gs_hash) << "\"," << std::endl;
    json << "  \"blend_enabled\": " << info.blend_enabled << "," << std::endl;
    json << "  \"textures\": [" << std::endl;

    for (size_t i = 0; i < images.size(); ++i) {
        const auto& [slot, image_id] = images[i];
        const auto& img = cache.GetImage(image_id);
        json << "    {" << std::endl;
        json << "      \"slot\": " << slot << "," << std::endl;
        json << "      \"width\": " << img.info.size.width << "," << std::endl;
        json << "      \"height\": " << img.info.size.height << "," << std::endl;
        json << "      \"format\": " << static_cast<u32>(img.info.pixel_format) << "," << std::endl;
        json << "      \"tile_mode\": " << static_cast<u32>(img.info.tile_mode) << "," << std::endl;
        json << "      \"mips\": " << img.info.resources.num_mips << "," << std::endl;
        json << "      \"gpu_addr\": \"0x" << fmt::format("{:016x}", img.info.guest_address) << "\"" << std::endl;
        json << "    }" << (i + 1 < images.size() ? "," : "") << std::endl;
    }

    json << "  ]" << std::endl;
    json << "}" << std::endl;
    json.close();

    // Dump raw textures
    for (size_t i = 0; i < images.size() && i < 8; ++i) {
        const auto& [slot, image_id] = images[i];
        const auto& img = cache.GetImage(image_id);
        std::string raw_path = fmt::format("{}/tex{:02d}_slot{:02d}_{}x{}.raw",
                                           dir, i, slot, img.info.size.width, img.info.size.height);
        // Note: actual GPU readback needs scheduler support — for now, log that we would dump
        LOG_DEBUG(Lib_GnmDriver,
                  "KNACK_TEXTURE_DUMP frame={} draw={} tex={} slot={} path={} "
                  "size={}x{} fmt={} gpu_addr=0x{:016x}",
                  g_frame_id.load(), info.draw_id, i, slot, raw_path,
                  img.info.size.width, img.info.size.height,
                  static_cast<u32>(img.info.pixel_format), img.info.guest_address);
    }

    LOG_DEBUG(Lib_GnmDriver,
              "KNACK_TEXTURE_DUMP_DONE frame={} draw={} type={} count={} path={}",
              g_frame_id.load(), info.draw_id, EffectTypeName(info.effect_type), count, dir);
}

// ─── FS Summary ────────────────────────────────────────────────────
FsSummary& FsSummary::Instance() {
    static FsSummary instance;
    return instance;
}

void FsSummary::RecordMissing(const std::string& path) {
    if (!g_flags.fs_summary) return;
    std::lock_guard lock(mtx);

    current_ts++;
    for (auto& entry : entries) {
        if (entry.path == path) {
            entry.count++;
            entry.last_ts = current_ts;
            return;
        }
    }

    FsMissingEntry e;
    e.path = path;
    e.count = 1;
    e.first_ts = current_ts;
    e.last_ts = current_ts;
    // Check characteristics
    e.is_csv = (path.find(".csv") != std::string::npos);
    e.is_hostapp = (path.find("/hostapp/") != std::string::npos ||
                    path.find("\\hostapp\\") != std::string::npos);
    entries.push_back(e);
}

void FsSummary::RecordCdataOpen(const std::string& path) {
    if (!g_flags.fs_summary) return;
    std::lock_guard lock(mtx);
    for (auto& entry : entries) {
        if (entry.is_csv || entry.is_hostapp) {
            entry.cdata_found = true;
        }
    }
    LOG_DEBUG(Lib_GnmDriver, "KNACK_FS_CDATA_OPEN path={}", path);
}

void FsSummary::WriteSummary(const std::string& filename) {
    if (!g_flags.fs_summary) return;
    std::lock_guard lock(mtx);

    if (entries.empty()) {
        LOG_DEBUG(Lib_GnmDriver, "KNACK_FS_SUMMARY: no missing files recorded");
        return;
    }

    std::ofstream out(filename);
    out << "# KNACK FS Missing Files Summary" << std::endl;
    out << "# Generated: " << std::time(nullptr) << std::endl;
    out << "# Total unique paths: " << entries.size() << std::endl;
    out << std::endl;
    out << "path\tcount\tfirst_ts\tlast_ts\tis_csv\tis_hostapp\tcdata_found" << std::endl;

    for (const auto& e : entries) {
        out << e.path << "\t" << e.count << "\t" << e.first_ts << "\t" << e.last_ts << "\t"
            << e.is_csv << "\t" << e.is_hostapp << "\t" << e.cdata_found << std::endl;
    }
    out.close();

    LOG_DEBUG(Lib_GnmDriver,
              "KNACK_FS_SUMMARY_WRITTEN path={} entries={}",
              filename, entries.size());
}

void KnackFsRecordMissing(const std::string& path) {
    FsSummary::Instance().RecordMissing(path);
}

void KnackFsRecordCdataOpen(const std::string& path) {
    FsSummary::Instance().RecordCdataOpen(path);
}

void KnackFsWriteSummary() {
    FsSummary::Instance().WriteSummary("dumps/KNACK_FS_MISSING_SUMMARY.txt");
}

} // namespace KnackDiag
