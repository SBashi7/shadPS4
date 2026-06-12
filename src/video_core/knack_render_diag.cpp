// SPDX-FileCopyrightText: Copyright 2025 KNACK Render Diagnostics
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/knack_render_diag.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <xxhash.h>
#include "common/logging/log.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/texture_cache/image.h"

namespace KnackDiag {

// ─── Global state ──────────────────────────────────────────────────
Flags g_flags;
std::atomic<u64> g_frame_id{0};
std::atomic<u64> g_submit_id{0};
std::atomic<u64> g_draw_id{0};
std::atomic<u32> g_texture_dump_count{0};
std::atomic<bool> g_current_frame_is_effect{false};

static std::once_flag init_once;

// ─── Flags ─────────────────────────────────────────────────────────
Flags Flags::LoadFromEnv() {
    Flags f;
    f.render_diag = EnvBool(ENV_RENDER_DIAG, false);
    if (f.render_diag) {
        f.effect_diag = EnvBool(ENV_EFFECT_DIAG, true);
        f.texture_dump = EnvBool(ENV_TEXTURE_DUMP, false);
        f.fs_summary = EnvBool(ENV_FS_SUMMARY, true);
        f.pm4_trace_effects = EnvBool(ENV_PM4_TRACE_EFFECTS, true);
        f.renderdoc_labels = EnvBool(ENV_RENDERDOC_LABELS, true);
        f.pipeline_blend_key_diag = EnvBool(ENV_PIPELINE_BLEND_KEY_DIAG, false);
        f.disable_effect_pipeline_reuse = EnvBool(ENV_DISABLE_EFFECT_PIPELINE_REUSE, false);
    }
    return f;
}

bool Flags::EnvBool(const char* name, bool def) {
    const char* val = std::getenv(name);
    if (!val) {
        return def;
    }
    std::string s(val);
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

void Initialize() {
    std::call_once(init_once, [] {
        g_flags = Flags::LoadFromEnv();
        if (g_flags.render_diag) {
            LOG_INFO(Common, "=== KNACK RENDER DIAGNOSTICS ENABLED ===");
            LOG_INFO(Common, "  effect_diag       = {}", g_flags.effect_diag);
            LOG_INFO(Common, "  texture_dump      = {}", g_flags.texture_dump);
            LOG_INFO(Common, "  fs_summary        = {}", g_flags.fs_summary);
            LOG_INFO(Common, "  pm4_trace_effects = {}", g_flags.pm4_trace_effects);
            LOG_INFO(Common, "  renderdoc_labels          = {}", g_flags.renderdoc_labels);
            LOG_INFO(Common, "  pipeline_blend_key_diag   = {}", g_flags.pipeline_blend_key_diag);
            LOG_INFO(Common, "  disable_effect_pipeline_reuse = {}",
                     g_flags.disable_effect_pipeline_reuse);
            LOG_INFO(Common, "=========================================");
            std::filesystem::create_directories("dumps/knack_effects");
            if (g_flags.fs_summary) {
                RegisterFsSummaryOnExit();
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
    case EffectType::ParticleTrail:
        return "PARTICLE_TRAIL";
    case EffectType::Explosion:
        return "EXPLOSION";
    case EffectType::SmokeCloud:
        return "SMOKE_CLOUD";
    case EffectType::Sunstone:
        return "SUNSTONE";
    case EffectType::TransparentEffect:
        return "TRANSPARENT_EFFECT";
    default:
        return "UNKNOWN";
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

    info.vs_hash = regs.vs_program.address;
    info.fs_hash = regs.ps_program.address;
    info.gs_hash = regs.gs_program.address;
    info.cs_hash = 0;

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

    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        info.write_mask[i] = regs.color_target_mask.GetMask(i);
    }

    info.depth_test = regs.depth_control.depth_enable;
    info.depth_write = regs.depth_control.depth_write_enable;

    info.num_indices = regs.num_indices;
    info.num_instances = regs.num_instances.NumInstances();

    info.has_geometry_shader = regs.stage_enable.gs_en != 0;

    info.prim_type = static_cast<u32>(regs.primitive_type);

    info.effect_type = info.Classify();
    return info;
}

bool EffectDrawInfo::IsEffect() const {
    return effect_type != EffectType::Unknown;
}

EffectType EffectDrawInfo::Classify() const {
    if (blend_enabled) {
        const bool is_additive =
            (color_src_factor == static_cast<u32>(AmdGpu::BlendControl::BlendFactor::SrcAlpha)) &&
            (color_dst_factor == static_cast<u32>(AmdGpu::BlendControl::BlendFactor::One));
        const bool is_alpha_blend =
            (color_src_factor == static_cast<u32>(AmdGpu::BlendControl::BlendFactor::SrcAlpha)) &&
            (color_dst_factor ==
             static_cast<u32>(AmdGpu::BlendControl::BlendFactor::OneMinusSrcAlpha));

        if (is_additive) {
            if (num_indices > 100) {
                return EffectType::Sunstone;
            }
            return EffectType::ParticleTrail;
        }
        if (is_alpha_blend) {
            if (has_geometry_shader) {
                return EffectType::ParticleTrail;
            }
            return EffectType::TransparentEffect;
        }
    }

    if (has_geometry_shader) {
        return EffectType::ParticleTrail;
    }

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
              submit_id, draw_id, EffectTypeName(effect_type), packet_offset, packet_index, vs_hash,
              fs_hash, gs_hash, blend_enabled, color_src_factor, color_dst_factor, color_func,
              alpha_src_factor, alpha_dst_factor, alpha_func, write_mask[0], write_mask[1],
              write_mask[2], write_mask[3], write_mask[4], write_mask[5], write_mask[6],
              write_mask[7], mrt_mask, depth_test, depth_write, has_geometry_shader, prim_type,
              num_indices, num_instances);
}

// ─── Texture descriptor logging ────────────────────────────────────
void LogBoundTexture(u32 slot, const VideoCore::Image& image, const char* prefix) {
    LOG_DEBUG(Lib_GnmDriver,
              "{} slot={} gpu_addr=0x{:016x} size={}x{}x{} fmt={} tile={} "
              "mips={} samples={} pitch={} usage=0x{:x} is_depth={}",
              prefix, slot, image.info.guest_address, image.info.size.width, image.info.size.height,
              image.info.size.depth, static_cast<u32>(image.info.pixel_format),
              static_cast<u32>(image.info.tile_mode), image.info.resources.levels,
              image.info.num_samples, image.info.pitch, static_cast<u32>(image.usage_flags),
              image.info.props.is_depth);
}

// ─── PM4 drift trace ───────────────────────────────────────────────
void LogPm4DriftContext(std::span<const u32> dcb, size_t current_offset, u32 header_raw, u32 type,
                       bool skipped) {
    if (!g_flags.pm4_trace_effects) {
        return;
    }

    const u32* data = dcb.data();
    const size_t remaining = dcb.size();

    LOG_DEBUG(Lib_GnmDriver, "KNACK_PM4_DRIFT off={} rem={} header=0x{:08x} type={} skipped={}",
              current_offset, remaining, header_raw, type, skipped);

    if (current_offset >= 1) {
        const size_t start = (current_offset >= 8) ? (current_offset - 8) : 0;
        const size_t n = std::min<size_t>(current_offset - start, 8);
        for (size_t i = 0; i < n; ++i) {
            LOG_DEBUG(Lib_GnmDriver, "KNACK_PM4_DRIFT_PREV[{}] = 0x{:08x}", i, data[start + i]);
        }
    }

    // Is this inside a NOP payload?
    if ((current_offset >= 1 && data[current_offset - 1] == 0xc0391000) ||
        (current_offset >= 2 && data[current_offset - 2] == 0xc0391000)) {
        LOG_DEBUG(Lib_GnmDriver, "KNACK_PM4_DRIFT_INSIDE_NOP_PAYLOAD offset={}", current_offset);
    }

    const size_t max_dump = std::min<size_t>(remaining, 8);
    for (size_t i = 0; i < max_dump; ++i) {
        LOG_DEBUG(Lib_GnmDriver, "KNACK_PM4_DRIFT_NEXT[{}] = 0x{:08x}", i,
                  data[current_offset + i]);
    }
}

// ─── FS Summary ────────────────────────────────────────────────────

namespace {
struct FsMissingEntry {
    std::string path;
    u32 count = 1;
    u32 first_ts = 0;
    u32 last_ts = 0;
    bool is_csv = false;
    bool is_hostapp = false;
    bool cdata_found = false;
};

class FsSummary {
public:
    static FsSummary& Instance() {
        static FsSummary instance;
        return instance;
    }

    void RecordMissing(const std::string& path) {
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
        e.is_csv = (path.find(".csv") != std::string::npos);
        e.is_hostapp = (path.find("/hostapp/") != std::string::npos ||
                        path.find("\\hostapp\\") != std::string::npos);
        entries.push_back(e);
    }

    void RecordCdataOpen() {
        std::lock_guard lock(mtx);
        for (auto& entry : entries) {
            if (entry.is_csv || entry.is_hostapp) {
                entry.cdata_found = true;
            }
        }
    }

    void WriteSummary(const std::string& filename) {
        std::lock_guard lock(mtx);
        if (entries.empty()) {
            return;
        }
        std::ofstream out(filename);
        out << "# KNACK FS Missing Files Summary\n";
        out << "# Generated: " << std::time(nullptr) << "\n";
        out << "# Total unique paths: " << entries.size() << "\n\n";
        out << "path\tcount\tfirst_ts\tlast_ts\tis_csv\tis_hostapp\tcdata_found\n";
        for (const auto& e : entries) {
            out << e.path << "\t" << e.count << "\t" << e.first_ts << "\t" << e.last_ts << "\t"
                << e.is_csv << "\t" << e.is_hostapp << "\t" << e.cdata_found << "\n";
        }
        out.close();
        LOG_DEBUG(Lib_GnmDriver, "KNACK_FS_SUMMARY_WRITTEN path={} entries={}", filename,
                  entries.size());
    }

private:
    std::mutex mtx;
    std::vector<FsMissingEntry> entries;
    u32 current_ts = 0;
};
} // namespace

void KnackFsRecordMissing(const std::string& path) {
    if (g_flags.fs_summary) {
        FsSummary::Instance().RecordMissing(path);
    }
}

void KnackFsRecordCdataOpen(const std::string& path) {
    if (g_flags.fs_summary) {
        (void)path;
        FsSummary::Instance().RecordCdataOpen();
    }
}

void KnackFsWriteSummary() {
    if (g_flags.fs_summary) {
        FsSummary::Instance().WriteSummary("dumps/KNACK_FS_MISSING_SUMMARY.txt");
    }
    // Also write effect signature summary
    if (g_flags.effect_diag) {
        EffectSignatureTracker::Instance().WriteSummary(
            "dumps/KNACK_EFFECT_SIGNATURE_SUMMARY.txt");
    }
}

void RegisterFsSummaryOnExit() {
    std::atexit([] { KnackFsWriteSummary(); });
}

// ─── Effect signature grouping ─────────────────────────────────────

EffectSignatureTracker& EffectSignatureTracker::Instance() {
    static EffectSignatureTracker instance;
    return instance;
}

void EffectSignatureTracker::RecordDraw(const EffectDrawInfo& info, u32 num_bound_images,
                                          u32 fmt_0, u32 tile_0, u32 fmt_1, u32 tile_1,
                                          bool pipe_blend_en, bool pipe_blend_en2,
                                          u32 pipe_write_mask, u32 reg_write_mask) {
    std::lock_guard lock(mtx);
    total_draws++;

    const auto hash = XXH3_64bits(&info.vs_hash, sizeof(info.vs_hash) * 2);
    EffectSignature sig{};
    sig.vs_hash = info.vs_hash;
    sig.fs_hash = info.fs_hash;
    sig.gs_hash = info.gs_hash;
    sig.effect_type = info.effect_type;
    sig.fmt_0 = fmt_0;
    sig.tile_0 = tile_0;
    sig.has_gs = info.has_geometry_shader;
    sig.pipe_blend_en = pipe_blend_en;
    sig.blend_mismatch = (pipe_blend_en != pipe_blend_en2);

    // Find or create entry
    for (auto& entry : entries) {
        if (entry.sig == sig) {
            entry.draw_count++;
            entry.last_submit = info.submit_id;
            entry.last_draw = info.draw_id;
            return;
        }
    }
    sig.draw_count = 1;
    sig.first_submit = info.submit_id;
    sig.first_draw = info.draw_id;
    entries.emplace_back(sig);
}

void EffectSignatureTracker::WriteSummary(const std::string& path) {
    std::lock_guard lock(mtx);
    if (entries.empty()) {
        return;
    }

    // Sort: mismatch first, then by count
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        if (a.blend_mismatch != b.blend_mismatch) {
            return a.blend_mismatch > b.blend_mismatch;
        }
        return a.draw_count > b.draw_count;
    });

    std::ofstream out(path);
    out << "# KNACK Effect Draw Signature Summary\n";
    out << "# Total draws: " << total_draws << "\n";
    out << "# Unique signatures: " << entries.size() << "\n";
    out << "# Sorted: mismatch first, then by draw count\n\n";
    out << "draw_count\tmismatch\tvs_hash\tfs_hash\tgs_hash\ttype\tfmt0\ttile0\thas_gs\tpipe_blend\t"
           "first_submit\tfirst_draw\n";

    const u32 limit = std::min<u32>(30, (u32)entries.size());
    for (u32 i = 0; i < limit; ++i) {
        const auto& e = entries[i];
        out << e.draw_count << "\t" << e.blend_mismatch << "\t"
            << fmt::format("0x{:016x}", e.vs_hash) << "\t"
            << fmt::format("0x{:016x}", e.fs_hash) << "\t"
            << fmt::format("0x{:016x}", e.gs_hash) << "\t"
            << EffectTypeName(e.effect_type) << "\t" << e.fmt_0 << "\t" << e.tile_0
            << "\t" << e.has_gs << "\t" << e.pipe_blend_en << "\t"
            << e.first_submit << "\t" << e.first_draw << "\n";
    }
    out.close();
    LOG_DEBUG(Lib_GnmDriver, "KNACK_EFFECT_SIGNATURE_SUMMARY path={} sigs={} draws={}", path,
              entries.size(), total_draws);
}

} // namespace KnackDiag
