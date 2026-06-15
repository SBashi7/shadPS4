// SPDX-FileCopyrightText: Copyright 2025 KNACK Render Diagnostics
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
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
inline constexpr auto ENV_TEXTURE_AUDIT = "KNACK_TEXTURE_AUDIT";
inline constexpr auto ENV_TEXTURE_DUMP_MAX = "KNACK_TEXTURE_DUMP_MAX";
inline constexpr auto ENV_TEXTURE_DUMP_SHADER = "KNACK_TEXTURE_DUMP_SHADER";
inline constexpr auto ENV_TEXTURE_DUMP_TOP_N = "KNACK_TEXTURE_DUMP_TOP_N";
inline constexpr auto ENV_FRAME_IMAGE_FORCE_SAFE_COPY = "KNACK_FRAME_IMAGE_FORCE_SAFE_COPY";
inline constexpr auto ENV_WATCH_ADDR = "KNACK_WATCH_ADDR";
inline constexpr auto ENV_WATCH_SIZE = "KNACK_WATCH_SIZE";
inline constexpr auto ENV_WRITER_AUDIT = "KNACK_WRITER_AUDIT";
inline constexpr auto ENV_WRITER_VS = "KNACK_WRITER_VS";
inline constexpr auto ENV_WRITER_FS = "KNACK_WRITER_FS";
inline constexpr auto ENV_WRITER_DUMP_MAX = "KNACK_WRITER_DUMP_MAX";
inline constexpr auto ENV_FORCE_REDETILE = "KNACK_FORCE_REDETILE_BEFORE_FINAL";

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
    bool texture_audit = false;       // Texture audit CSV
    u32 texture_dump_max = 20;        // Max dumps per run
    u64 texture_dump_shader = 0;      // Only dump for this specific shader hash (0=top auto)
    u32 texture_dump_top_n = 2;
    bool frame_image_force_safe_copy = false;
    u64 watch_addr = 0x2a8ea0000;
    u32 watch_size = 0xE10000;
    bool force_redetile = false;
    bool writer_audit = false;
    u64 writer_vs = 0x029cf6b4;
    u64 writer_fs = 0x029cf6b8;
    u32 writer_dump_max = 10;
    bool writer_slot2_force_unorm = false;
    bool writer_slot2_force_snorm = false;
    bool skip_shader_292ecf7 = false;
    s32 zero_slot_292ecf7 = -1;

    // Tornado capture
    bool tornado_capture = false;  // Hardcoded ON for capture build

    static Flags LoadFromEnv();

    [[nodiscard]] bool AnyEffectDiag() const {
        return effect_diag || texture_dump || pm4_trace_effects || renderdoc_labels || texture_audit;
    }

private:
    static bool EnvBool(const char* name, bool def);
    static u32 EnvU32(const char* name, u32 def);
    static u64 EnvU64(const char* name, u64 def);
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

// ─── Texture Audit ─────────────────────────────────────────────────

struct TexAuditEntry {
    u64 frame;
    u64 draw;
    u64 submit;
    const char* effect_category;
    u64 vs_hash, fs_hash, cs_hash;
    u32 pipeline_id;
    u32 rt_fmt;
    u32 rt_w, rt_h;
    u32 tex_count;
    u32 img_id;
    u64 gpu_addr;
    u32 width, height, depth;
    u32 pitch;
    u32 mips;
    u32 data_fmt, num_fmt;
    u32 vk_fmt;
    bool is_srgb;
    u32 tile_mode;
    u32 array_mode;
    bool is_tiled;
    u32 usage_flags;
    bool is_fullscreen_rt;
};

class TextureAuditWriter {
public:
    static TextureAuditWriter& Instance();
    void RecordEffectDraw(const TexAuditEntry& entry);
    void Flush();

private:
    std::mutex mtx;
    std::vector<TexAuditEntry> entries;
    std::chrono::steady_clock::time_point last_flush;
    static constexpr u32 FLUSH_INTERVAL = 250; // draws
    static constexpr u32 FLUSH_INTERVAL_SEC = 5;
};

// Called from vk_rasterizer for each suspect effect draw
void TextureAuditRecord(const TexAuditEntry& entry);

// ─── Targeted Texture Dump ─────────────────────────────────────────

class TextureDumpManager {
public:
    static TextureDumpManager& Instance();
    void RegisterSuspect(u64 vs_hash, u64 fs_hash, u64 draw_id, u64 gpu_addr, u32 size_bytes);
    bool ShouldDump(u64 vs_hash, u64 fs_hash) const;
    bool CanDump() const; // respects max dump limit
    void RecordDump();
    const std::string& GetDumpDir() const;

private:
    std::mutex mtx;
    std::map<u64, u32> shader_suspect_count; // hash -> count
    u32 total_dumps = 0;
    std::string dump_dir;
};

void TextureDumpRecordSuspect(u64 vs, u64 fs, u64 draw, u64 addr, u32 bytes);

// ─── Frame Image Producer Tracking ─────────────────────────────────

enum class FrameImageWriteType : u32 {
    Unknown = 0,
    ColorAttachment,
    StorageImage,
    TransferDst,
    Copy,
    Blit,
    Resolve,
    Clear,
    InitialContents,
};

struct FrameImageWrite {
    u64 frame;
    u64 submit;
    u64 draw;
    u32 image_id;
    FrameImageWriteType type;
    u64 vs_hash;
    u64 fs_hash;
    u64 gpu_addr;
    const char* vk_format_name;
    u32 width, height;
};

class FrameImageTracker {
public:
    static FrameImageTracker& Instance();
    void RecordImageCreate(u32 image_id, u64 gpu_addr, u32 width, u32 height, u32 vk_fmt,
                           bool is_tiled, bool is_depth);
    void RecordImageWrite(u32 image_id, u64 gpu_addr, FrameImageWriteType type);
    void RecordFinalCompositeSample(u32 image_id, u64 gpu_addr, u64 draw_id, u64 vs_hash,
                                    u64 fs_hash);
    void SetForceSafeCopy(bool val);

private:
    std::mutex mtx;
    std::unordered_map<u64, FrameImageWrite> last_writes; // gpu_addr -> last write
    std::ofstream csv;
    bool csv_header_written = false;
    void EnsureCsv();
    void FlushCsv(const FrameImageWrite& entry, const char* marker);
};

// Hook functions called from render pipeline
void FrameImageRecordCreate(u32 img_id, u64 gpu_addr, u32 w, u32 h, u32 vk_fmt, bool tiled,
                            bool depth);
void FrameImageRecordWrite(u32 img_id, u64 gpu_addr, FrameImageWriteType type);
void FrameImageRecordFinalSample(u32 img_id, u64 gpu_addr, u64 draw, u64 vs, u64 fs);

// ─── Memory Range Watch ────────────────────────────────────────────

class MemoryWatcher {
public:
    static MemoryWatcher& Instance();
    void Init(u64 addr, u32 size);
    u64 HashGuestMemory(u64 addr, u32 size);
    void RecordDetile(u64 addr);
    void RecordFinalSample(u64 addr, u64 draw_id);
    void RecordWrite(u64 addr, const char* source);

private:
    std::mutex mtx;
    u64 watch_addr = 0;
    u32 watch_size = 0;
    u64 last_detile_hash = 0;
    u64 last_detile_frame = 0;
    u32 detile_count = 0;
    u32 final_sample_count = 0;
    std::map<u64, u32> final_sample_addrs;
public:
    u32 GetDetileCount() const { return detile_count; }
    u32 GetFinalSampleCount() const { return final_sample_count; }
};

void WatchRecordDetile(u64 addr);
void WatchRecordFinalSample(u64 addr, u64 draw);
void WatchRecordWrite(u64 addr, const char* source);

// ─── Vulkan Image Writer Tracking ──────────────────────────────────

void VkImageRecordColorWrite(u64 gpu_addr, u64 vs_hash, u64 fs_hash,
                             u32 wmask, bool blend, u32 rt_fmt, u32 depth_en,
                             u32 depth_write, u32 color_export, u32 num_tex);
void VkImageRecordComputeWrite(u64 gpu_addr, u64 cs_hash);
void VkImageRecordFinalSample(u64 gpu_addr);
const char* VkImageGetLastWriter(u64 gpu_addr);

// ─── Tornado Capture System ────────────────────────────────────────

class TornadoCapture {
public:
    static TornadoCapture& Instance();
    void CheckTrigger();  // Called each frame to check for trigger file
    bool IsCapturing() const;
    void RecordDraw(u64 vs_hash, u64 fs_hash, u64 gs_hash, u64 cs_hash,
                    u32 num_idx, u32 num_inst, u64 output_addr, u32 output_fmt,
                    u32 wmask, bool blend, u32 depth_en, u32 depth_write,
                    u32 color_export, u32 shader_mask, u32 target_mask);
    void RecordImageSample(u32 slot, u64 addr, u32 fmt, u32 tile, u32 w, u32 h,
                           bool same_output, const char* writer);
    void EndFrame();
    u64 HashGuest(u64 addr, u32 size);
    void DumpBmp(const std::string& path, u64 addr, u32 w, u32 h);

private:
    std::mutex mtx;
    bool active = false;
    u32 capture_frame = 0;
    u32 total_frames = 0;
    static constexpr u32 CAPTURE_MAX = 60; // 60 frames ≈ 3 seconds
    std::ofstream csv;
    std::map<std::pair<u32, u32>, u32> shader_counts;
    void DumpShaderSummary();
};

void TornadoRecordDraw(u64 vs, u64 fs, u64 gs, u64 cs, u32 idx, u32 inst,
                       u64 out_addr, u32 out_fmt, u32 wmask, bool blend,
                       u32 depth_en, u32 depth_write, u32 color_exp,
                       u32 shader_mask, u32 target_mask);
void TornadoRecordSample(u32 slot, u64 addr, u32 fmt, u32 tile, u32 w, u32 h,
                         bool same_out, const char* writer);
// ─── Runtime Shader Test System ─────────────────────────────────────

enum class ShaderTestMode : u8 { Noop, Skip, Black, Alpha0 };

struct ShaderTestRule {
    u64 vs_hash;
    u64 fs_hash;
    ShaderTestMode mode;
};

class ShaderTestSystem {
public:
    struct RuleHit {
        u64 vs, fs;
        ShaderTestMode mode;
        u32 count = 0;
    };

    static ShaderTestSystem& Instance();
    void CheckReload();
    ShaderTestMode GetMode(u64 vs, u64 fs) const;
    const char* GetModeStr(ShaderTestMode m) const;

    std::vector<RuleHit> rule_hits;
    void LogRuleHit(u64 vs, u64 fs, ShaderTestMode mode);

private:
    std::mutex mtx;
    std::vector<ShaderTestRule> rules;
    std::chrono::steady_clock::time_point last_check;
};

// Called from vk_rasterizer to check if draw should be skipped
bool ShaderTestShouldSkip(u64 vs, u64 fs);
ShaderTestMode ShaderTestGetMode(u64 vs, u64 fs);
bool WatchShouldForceRedetile(u64 addr);

} // namespace KnackDiag
