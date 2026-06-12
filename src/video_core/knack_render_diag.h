// SPDX-FileCopyrightText: Copyright 2025 KNACK Render Diagnostics
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "common/types.h"

// Forward declarations — minimal to avoid type conflicts
namespace AmdGpu {
union Regs;
}

namespace VideoCore {
class Image;
} // namespace VideoCore

// ─── Master enable ──────────────────────────────────────────────────
// KNACK_RENDER_DIAG=1  → enables all diag subset flags below
// Individual flags can override: KNACK_EFFECT_DIAG=0 disables a subset

namespace KnackDiag {

// ─── Environment variable names ─────────────────────────────────────
inline constexpr auto ENV_RENDER_DIAG = "KNACK_RENDER_DIAG";
inline constexpr auto ENV_EFFECT_DIAG = "KNACK_EFFECT_DIAG";
inline constexpr auto ENV_TEXTURE_DUMP = "KNACK_TEXTURE_DUMP";
inline constexpr auto ENV_FS_SUMMARY = "KNACK_FS_SUMMARY";
inline constexpr auto ENV_PM4_TRACE_EFFECTS = "KNACK_PM4_TRACE_EFFECTS";
inline constexpr auto ENV_RENDERDOC_LABELS = "KNACK_RENDERDOC_LABELS";
inline constexpr auto ENV_PIPELINE_BLEND_KEY_DIAG = "KNACK_PIPELINE_BLEND_KEY_DIAG";
inline constexpr auto ENV_DISABLE_EFFECT_PIPELINE_REUSE = "KNACK_DISABLE_EFFECT_PIPELINE_REUSE";

// ─── Feature flags ──────────────────────────────────────────────────
struct Flags {
    bool render_diag = false;
    bool effect_diag = false;
    bool texture_dump = false;
    bool fs_summary = false;
    bool pm4_trace_effects = false;
    bool renderdoc_labels = false;
    bool pipeline_blend_key_diag = false;
    bool disable_effect_pipeline_reuse = false;

    static Flags LoadFromEnv();

    [[nodiscard]] bool AnyEffectDiag() const {
        return effect_diag || texture_dump || pm4_trace_effects || renderdoc_labels;
    }

private:
    static bool EnvBool(const char* name, bool def);
};

// ─── Global state (initialized once at startup) ─────────────────────
extern Flags g_flags;
extern std::atomic<u64> g_frame_id;
extern std::atomic<u64> g_submit_id;
extern std::atomic<u64> g_draw_id;
extern std::atomic<u32> g_texture_dump_count;
extern std::atomic<bool> g_current_frame_is_effect;
static constexpr u32 MAX_TEXTURE_DUMPS = 20;

void Initialize();
bool IsEnabled();
const Flags& GetFlags();

// ─── Effect detection heuristics ────────────────────────────────────

enum class EffectType : u32 {
    Unknown = 0,
    ParticleTrail,
    Explosion,
    SmokeCloud,
    Sunstone,
    TransparentEffect,
};

const char* EffectTypeName(EffectType t);

struct EffectDrawInfo {
    u64 submit_id;
    u64 draw_id;
    size_t packet_offset;
    u32 packet_index;
    EffectType effect_type;
    u64 vs_hash;
    u64 fs_hash;
    u64 gs_hash;
    u64 cs_hash;
    u32 pipeline_id;
    bool blend_enabled;
    u32 color_src_factor;
    u32 color_func;
    u32 color_dst_factor;
    u32 alpha_src_factor;
    u32 alpha_func;
    u32 alpha_dst_factor;
    u32 write_mask[8];
    u32 mrt_mask;
    bool depth_test;
    bool depth_write;
    u32 num_indices;
    u32 num_instances;
    bool has_geometry_shader;
    u32 prim_type;

    // Implementation in .cpp to avoid heavy includes
    static EffectDrawInfo FromRegs(const AmdGpu::Regs& regs, u64 submit, u64 draw, size_t pkt_off,
                                   u32 pkt_idx);
    bool IsEffect() const;
    EffectType Classify() const;
    void Log() const;
};

// ─── Texture descriptor logging ────────────────────────────────────

void LogBoundTexture(u32 slot, const VideoCore::Image& image, const char* prefix);

// ─── PM4 drift trace ───────────────────────────────────────────────

void LogPm4DriftContext(std::span<const u32> dcb, size_t current_offset, u32 header_raw, u32 type,
                        bool skipped);

// ─── FS summary ────────────────────────────────────────────────────

void KnackFsRecordMissing(const std::string& path);
void KnackFsRecordCdataOpen(const std::string& path);
void KnackFsWriteSummary();

// ─── Convenience: register FS summary on exit ───────────────────────

void RegisterFsSummaryOnExit();

// ─── Effect signature grouping ─────────────────────────────────────

struct EffectSignature {
    u64 vs_hash;
    u64 fs_hash;
    u64 gs_hash;
    EffectType effect_type;
    u32 fmt_0;
    u32 tile_0;
    bool has_gs;
    bool pipe_blend_en;
    bool blend_mismatch;
    u32 draw_count;
    u64 first_submit;
    u64 first_draw;
    u64 last_submit;
    u64 last_draw;

    bool operator==(const EffectSignature& o) const {
        return vs_hash == o.vs_hash && fs_hash == o.fs_hash && gs_hash == o.gs_hash &&
               effect_type == o.effect_type && fmt_0 == o.fmt_0 && tile_0 == o.tile_0 &&
               has_gs == o.has_gs && pipe_blend_en == o.pipe_blend_en &&
               blend_mismatch == o.blend_mismatch;
    }
};

class EffectSignatureTracker {
public:
    static EffectSignatureTracker& Instance();
    void RecordDraw(const EffectDrawInfo& info, u32 num_bound_images, u32 fmt_0, u32 tile_0,
                    u32 fmt_1, u32 tile_1, bool pipe_blend_en, bool reg_blend_en,
                    u32 pipe_write_mask, u32 reg_write_mask);
    void WriteSummary(const std::string& path);

private:
    std::mutex mtx;
    std::vector<EffectSignature> entries;
    u64 total_draws = 0;
};

} // namespace KnackDiag
