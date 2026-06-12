// SPDX-FileCopyrightText: Copyright 2025 KNACK Render Diagnostics
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include "common/types.h"
#include "common/logging/log.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/pm4_cmds.h"

// Forward declarations
namespace VideoCore {
struct ImageId;
class Image;
class TextureCache;
} // namespace VideoCore

namespace Vulkan {
class Rasterizer;
class GraphicsPipeline;
class ComputePipeline;
} // namespace Vulkan

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

// ─── Feature flags ──────────────────────────────────────────────────
struct Flags {
    bool render_diag = false;       // Master enable
    bool effect_diag = false;       // Log effect draws
    bool texture_dump = false;      // Dump bound textures for effect draws
    bool fs_summary = false;        // Aggregate FS misses
    bool pm4_trace_effects = false; // PM4 drift trace around effects
    bool renderdoc_labels = false;  // Vulkan debug labels for RenderDoc

    static Flags LoadFromEnv() {
        Flags f;
        f.render_diag = EnvBool(ENV_RENDER_DIAG, false);
        if (f.render_diag) {
            // Master enable: all sub-flags default on unless explicitly set to 0
            f.effect_diag = EnvBool(ENV_EFFECT_DIAG, true);
            f.texture_dump = EnvBool(ENV_TEXTURE_DUMP, false); // Dump OFF by default (large output)
            f.fs_summary = EnvBool(ENV_FS_SUMMARY, true);
            f.pm4_trace_effects = EnvBool(ENV_PM4_TRACE_EFFECTS, true);
            f.renderdoc_labels = EnvBool(ENV_RENDERDOC_LABELS, true);
        }
        return f;
    }

    [[nodiscard]] bool AnyEffectDiag() const { return effect_diag || texture_dump || pm4_trace_effects || renderdoc_labels; }

private:
    static bool EnvBool(const char* name, bool def) {
        const char* val = std::getenv(name);
        if (!val) return def;
        std::string s(val);
        return s == "1" || s == "true" || s == "yes" || s == "on";
    }
};

// ─── Global state (initialized once at startup) ─────────────────────
extern Flags g_flags;              // Set at first access
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
    ParticleTrail,     // Ground trails from character movement
    Explosion,          // Enemy defeat / breakable explosions
    SmokeCloud,         // Smoke / dust
    Sunstone,           // Sunstone ability effect
    TransparentEffect,  // Generic alpha-blended effect
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
    // Blend state
    bool blend_enabled;
    u32 color_src_factor;
    u32 color_func;
    u32 color_dst_factor;
    u32 alpha_src_factor;
    u32 alpha_func;
    u32 alpha_dst_factor;
    // Color write mask per RT
    u32 write_mask[8];
    u32 mrt_mask;
    // Depth state
    bool depth_test;
    bool depth_write;
    // Num instances / indices (for billboard detection)
    u32 num_indices;
    u32 num_instances;
    // Geometry shader presence
    bool has_geometry_shader;
    // Primitive type
    u32 prim_type;

    static EffectDrawInfo FromRegs(const AmdGpu::Regs& regs, u64 submit, u64 draw,
                                    size_t pkt_off, u32 pkt_idx);
    bool IsEffect() const;
    EffectType Classify() const;
    void Log() const;
};

// ─── Texture descriptor logging ────────────────────────────────────

struct TextureDiagInfo {
    u32 image_id;
    u64 gpu_address;
    u32 width, height, depth;
    u32 pitch;
    u32 pixel_format;     // Vulkan format
    u32 tile_mode;        // AmdGpu::TileMode
    u32 array_mode;
    u32 mip_count;
    u32 num_samples;
    bool is_srgb;
    bool is_tiled;
    bool is_depth;
    u32 usage_flags;      // vk::ImageUsageFlags

    void Log(const char* prefix) const;
};

void LogBoundTexture(u32 slot, const VideoCore::Image& image, const char* prefix = "KNACK_TEX");
void LogAllBoundTextures(u32 first_image_idx, u32 image_count, VideoCore::TextureCache& cache);

// ─── PM4 drift trace ───────────────────────────────────────────────

struct Pm4DriftEntry {
    u32 type;
    u32 header;
    size_t offset;
    size_t remaining;
    u32 prev_8[8];
    u32 next_8[8];
    bool is_nop_payload;
    bool skipped_as_default;
};

void LogPm4DriftContext(std::span<const u32> dcb, size_t current_offset,
                         u32 header_raw, u32 type, bool skipped);

// ─── Texture dumping ───────────────────────────────────────────────

void DumpEffectTextures(const EffectDrawInfo& info, VideoCore::TextureCache& cache,
                         const std::vector<std::pair<u32, VideoCore::ImageId>>& images);

// ─── FS summary ────────────────────────────────────────────────────

struct FsMissingEntry {
    std::string path;
    u32 count = 0;
    u32 first_ts = 0;
    u32 last_ts = 0;
    bool is_csv = false;
    bool is_hostapp = false;
    bool cdata_found = false;
};

class FsSummary {
public:
    static FsSummary& Instance();
    void RecordMissing(const std::string& path);
    void RecordCdataOpen(const std::string& path);
    void WriteSummary(const std::string& filename);

private:
    std::mutex mtx;
    std::vector<FsMissingEntry> entries;
    u32 current_ts = 0;
};

void KnackFsRecordMissing(const std::string& path);
void KnackFsRecordCdataOpen(const std::string& path);
void KnackFsWriteSummary();

} // namespace KnackDiag
