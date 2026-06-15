// SPDX-FileCopyrightText: Copyright 2025 KNACK Render Diagnostics
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/knack_render_diag.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
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
        // KNACK TEXTURE AUDIT: OFF by default. Set KNACK_TEXTURE_AUDIT=1 to enable.
        f.texture_audit = EnvBool(ENV_TEXTURE_AUDIT, false);
    if (f.texture_audit) {
        f.effect_diag = true;
        f.texture_dump = false;
        f.texture_dump_max = 20;
        f.texture_dump_shader = 0;
        f.texture_dump_top_n = 2;
        f.renderdoc_labels = false;
        f.pm4_trace_effects = false;
        f.fs_summary = false;
    }

    // Read watch config
    g_flags.watch_addr = EnvU64(ENV_WATCH_ADDR, 0x2a8ea0000);
    g_flags.watch_size = EnvU32(ENV_WATCH_SIZE, 0xE10000);
    f.force_redetile = false; // DISABLED: ForceUploadImage mid-frame breaks rendering
    MemoryWatcher::Instance().Init(g_flags.watch_addr, g_flags.watch_size);

    // KNACK TEXTURE AUDIT: OFF by default
    f.texture_audit = EnvBool(ENV_TEXTURE_AUDIT, false);
    if (f.texture_audit) {
        f.effect_diag = true;
    }

    // NEVER write files during startup. FrameImageTracker only uses LOG_INFO.
    // KNACK WRITER AUDIT: hardcoded ON for diagnostic
    f.writer_audit = true;
    f.writer_vs = 0x029cf6b4;
    f.writer_fs = 0x029cf6b8;
    f.writer_dump_max = 10;
    f.writer_slot2_force_snorm = false; // DISABLED: breaks menu, doesn't fix particles
    f.skip_shader_292ecf7 = false; // DISABLED
    f.zero_slot_292ecf7 = -1;
    f.tornado_capture = true; // HARDCODE: tornado capture ON

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

u32 Flags::EnvU32(const char* name, u32 def) {
    const char* val = std::getenv(name);
    if (!val) return def;
    return static_cast<u32>(std::strtoul(val, nullptr, 0));
}

u64 Flags::EnvU64(const char* name, u64 def) {
    const char* val = std::getenv(name);
    if (!val) return def;
    return std::strtoull(val, nullptr, 0);
}

void Initialize() {
    std::call_once(init_once, [] {
        g_flags = Flags::LoadFromEnv();
        LOG_INFO(Common, "=== KNACK DIAG: texture_audit={} effect_diag={} render={} ===",
                 g_flags.texture_audit, g_flags.effect_diag, g_flags.render_diag);
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
        if (entry == sig) {
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

void TextureAuditRecord(const TexAuditEntry& entry) {
    TextureAuditWriter::Instance().RecordEffectDraw(entry);
}

// ─── Texture audit writer implementation ───────────────────────────

TextureAuditWriter& TextureAuditWriter::Instance() {
    static TextureAuditWriter instance;
    return instance;
}

void TextureAuditWriter::RecordEffectDraw(const TexAuditEntry& entry) {
    // No file I/O — LOG_INFO only via the existing KNACK_EFFECT_DRAW logs
}

void TextureAuditWriter::Flush() {
    if (entries.empty()) return;

    std::ofstream csv("dumps/knack_texture_audit.csv", std::ios::app);
    for (const auto& e : entries) {
        csv << e.frame << "," << e.draw << "," << e.submit << "," << e.effect_category << ","
            << fmt::format("0x{:016x}", e.vs_hash) << "," << fmt::format("0x{:016x}", e.fs_hash)
            << "," << fmt::format("0x{:016x}", e.cs_hash) << "," << e.pipeline_id << ","
            << e.rt_fmt << "," << e.rt_w << "," << e.rt_h << "," << e.tex_count << "," << e.img_id
            << "," << fmt::format("0x{:016x}", e.gpu_addr) << "," << e.width << "," << e.height
            << "," << e.depth << "," << e.pitch << "," << e.mips << "," << e.data_fmt << ","
            << e.num_fmt << "," << e.vk_fmt << "," << e.is_srgb << "," << e.tile_mode << ","
            << e.array_mode << "," << e.is_tiled << "," << fmt::format("0x{:x}", e.usage_flags)
            << "," << e.is_fullscreen_rt << "\n";
    }
    csv.close();

    LOG_DEBUG(Lib_GnmDriver, "KNACK_TEXTURE_AUDIT_FLUSH entries={}", entries.size());
    entries.clear();
    last_flush = std::chrono::steady_clock::now();
}

// ─── Targeted texture dump ─────────────────────────────────────────

TextureDumpManager& TextureDumpManager::Instance() {
    static TextureDumpManager instance;
    return instance;
}

void TextureDumpManager::RegisterSuspect(u64 vs_hash, u64 fs_hash, u64 draw_id, u64 gpu_addr,
                                          u32 size_bytes) {
    std::lock_guard lock(mtx);
    // Count by combined hash
    u64 combined = vs_hash ^ (fs_hash << 32);
    shader_suspect_count[combined]++;
}

bool TextureDumpManager::ShouldDump(u64 vs_hash, u64 fs_hash) const {
    if (!g_flags.texture_dump) return false;

    // If specific shader requested
    if (g_flags.texture_dump_shader != 0) {
        return vs_hash == g_flags.texture_dump_shader || fs_hash == g_flags.texture_dump_shader;
    }

    // Auto: dump top N
    u64 combined = vs_hash ^ (fs_hash << 32);
    // Compare against top N (rough implementation)
    // For now, just allow until we have enough data
    return CanDump();
}

bool TextureDumpManager::CanDump() const {
    return total_dumps < g_flags.texture_dump_max;
}

void TextureDumpManager::RecordDump() {
    total_dumps++;
}

const std::string& TextureDumpManager::GetDumpDir() const {
    return dump_dir;
}

void TextureDumpRecordSuspect(u64 vs, u64 fs, u64 draw, u64 addr, u32 bytes) {
    TextureDumpManager::Instance().RegisterSuspect(vs, fs, draw, addr, bytes);
}

// ─── Frame Image Producer Tracking ─────────────────────────────────

FrameImageTracker& FrameImageTracker::Instance() {
    static FrameImageTracker instance;
    return instance;
}

void FrameImageTracker::EnsureCsv() {
    // No file I/O during startup — everything via LOG_INFO only
}

void FrameImageTracker::FlushCsv(const FrameImageWrite& entry, const char* marker) {
    // LOG_INFO only, no file I/O
}

void FrameImageTracker::RecordImageCreate(u32 image_id, u64 gpu_addr, u32 width, u32 height,
                                          u32 vk_fmt, bool is_tiled, bool is_depth) {
    // Only track fullscreen-ish images
    if (width < 1280 || height < 720) return;

    FrameImageWrite entry{};
    entry.frame = g_frame_id.load();
    entry.submit = 0;
    entry.draw = 0;
    entry.image_id = image_id;
    entry.type = FrameImageWriteType::InitialContents;
    entry.gpu_addr = gpu_addr;
    entry.width = width;
    entry.height = height;
    // Vulkan format name
    switch (vk_fmt) {
    case 37:
        entry.vk_format_name = "R8G8B8A8_UNORM";
        break;
    case 44:
        entry.vk_format_name = "B8G8R8A8_UNORM";
        break;
    default:
        entry.vk_format_name = "UNKNOWN";
        break;
    }

    FlushCsv(entry, "KNACK_FRAME_IMAGE_CREATE");

    if (is_tiled) {
        LOG_INFO(Render_Vulkan,
                 "KNACK_FRAME_IMAGE_CREATE id={} addr=0x{:016x} {}x{} fmt={} tiled={} depth={}",
                 image_id, gpu_addr, width, height, entry.vk_format_name, is_tiled, is_depth);
    }
}

void FrameImageTracker::RecordImageWrite(u32 image_id, u64 gpu_addr, FrameImageWriteType type) {
    FrameImageWrite entry{};
    entry.frame = g_frame_id.load();
    entry.submit = g_submit_id.load();
    entry.draw = g_draw_id.load();
    entry.image_id = image_id;
    entry.type = type;
    entry.gpu_addr = gpu_addr;

    const char* marker = "KNACK_FRAME_IMAGE_LAST_WRITE";
    switch (type) {
    case FrameImageWriteType::ColorAttachment:
        marker = "KNACK_COLOR_FULLSCREEN_WRITE";
        break;
    case FrameImageWriteType::StorageImage:
        marker = "KNACK_COMPUTE_FULLSCREEN_WRITE";
        break;
    case FrameImageWriteType::Copy:
    case FrameImageWriteType::TransferDst:
        marker = "KNACK_COPY_OR_RESOLVE_FULLSCREEN_WRITE";
        break;
    default:
        break;
    }
    FlushCsv(entry, marker);
}

void FrameImageTracker::RecordFinalCompositeSample(u32 image_id, u64 gpu_addr, u64 draw_id,
                                                    u64 vs_hash, u64 fs_hash) {
    FrameImageWrite entry{};
    entry.frame = g_frame_id.load();
    entry.submit = g_submit_id.load();
    entry.draw = draw_id;
    entry.image_id = image_id;
    entry.gpu_addr = gpu_addr;
    entry.vs_hash = vs_hash;
    entry.fs_hash = fs_hash;

    // Check if we know the last writer
    auto it = last_writes.find(gpu_addr);
    if (it != last_writes.end()) {
        LOG_INFO(Render_Vulkan,
                 "KNACK_FRAME_IMAGE_SAMPLED_FINAL draw={} addr=0x{:016x} last_writer=[{}] "
                 "last_frame={} last_submit={}",
                 draw_id, gpu_addr, static_cast<u32>(it->second.type), it->second.frame,
                 it->second.submit);
    } else {
        LOG_INFO(Render_Vulkan,
                 "KNACK_FRAME_IMAGE_UNKNOWN_PRODUCER draw={} addr=0x{:016x} "
                 "image_id={}",
                 draw_id, gpu_addr, image_id);
    }
    FlushCsv(entry, "KNACK_FRAME_IMAGE_SAMPLED_FINAL");
}

void FrameImageTracker::SetForceSafeCopy(bool val) {
    g_flags.frame_image_force_safe_copy = val;
}

void FrameImageRecordCreate(u32 img_id, u64 gpu_addr, u32 w, u32 h, u32 vk_fmt, bool tiled,
                            bool depth) {
    if (g_flags.render_diag || g_flags.effect_diag || g_flags.texture_audit) {
        FrameImageTracker::Instance().RecordImageCreate(img_id, gpu_addr, w, h, vk_fmt, tiled,
                                                         depth);
    }
}

void FrameImageRecordWrite(u32 img_id, u64 gpu_addr, FrameImageWriteType type) {
    if (g_flags.render_diag || g_flags.effect_diag || g_flags.texture_audit) {
        FrameImageTracker::Instance().RecordImageWrite(img_id, gpu_addr, type);
    }
}

void FrameImageRecordFinalSample(u32 img_id, u64 gpu_addr, u64 draw, u64 vs, u64 fs) {
    if (!g_flags.render_diag && !g_flags.effect_diag && !g_flags.texture_audit) return;
    FrameImageTracker::Instance().RecordFinalCompositeSample(img_id, gpu_addr, draw, vs, fs);
}

void WatchRecordDetile(u64 addr) { MemoryWatcher::Instance().RecordDetile(addr); }
void WatchRecordFinalSample(u64 addr, u64 draw) { MemoryWatcher::Instance().RecordFinalSample(addr, draw); }
void WatchRecordWrite(u64 addr, const char* source) { MemoryWatcher::Instance().RecordWrite(addr, source); }

// ─── MemoryWatcher ─────────────────────────────────────────────────

MemoryWatcher& MemoryWatcher::Instance() {
    static MemoryWatcher instance;
    return instance;
}

void MemoryWatcher::Init(u64 addr, u32 size) {
    watch_addr = addr;
    watch_size = size;
    LOG_INFO(Render_Vulkan, "KNACK_WATCH_INIT addr=0x{:016x} size=0x{:x}", addr, size);
}

u64 MemoryWatcher::HashGuestMemory(u64 addr, u32 size) {
    // Try to hash guest memory via buffer cache
    // For now, returns 0 (no access to mapped memory in this context)
    // The actual hashing is done in tile_manager.cpp which has buffer access
    return 0;
}

void MemoryWatcher::RecordDetile(u64 addr) {
    std::lock_guard lock(mtx);
    if (addr != watch_addr) return;

    detile_count++;
    last_detile_frame = g_frame_id.load();
    LOG_INFO(Render_Vulkan,
             "KNACK_WATCH_DETILE addr=0x{:016x} detile_count={} final_sample_count={} frame={}",
             addr, detile_count, final_sample_count, last_detile_frame);
}

void MemoryWatcher::RecordFinalSample(u64 addr, u64 draw_id) {
    std::lock_guard lock(mtx);
    if (addr != watch_addr) return;

    final_sample_count++;
    final_sample_addrs[addr]++;

    bool is_stale = (final_sample_count > detile_count);
    u64 current_frame = g_frame_id.load();

    LOG_INFO(Render_Vulkan,
             "KNACK_WATCH_FINAL_SAMPLE addr=0x{:016x} detile_count={} final_sample_count={} "
             "stale_cache={} force_redetile={}",
             addr, detile_count, final_sample_count, is_stale, g_flags.force_redetile);
}

void MemoryWatcher::RecordWrite(u64 addr, const char* source) {
    std::lock_guard lock(mtx);
    if (addr < watch_addr || addr >= watch_addr + watch_size) return;

    LOG_INFO(Render_Vulkan, "KNACK_WATCH_WRITE addr=0x{:016x} source={} frame={}",
             addr, source, g_frame_id.load());
}

bool WatchShouldForceRedetile(u64 addr) {
    if (!g_flags.force_redetile) return false;
    auto& w = MemoryWatcher::Instance();
    return (addr == g_flags.watch_addr) && (w.GetFinalSampleCount() > w.GetDetileCount());
}

// ─── Vulkan Image Writer Tracking ──────────────────────────────────

namespace {
struct VkImageWriter {
    const char* writer_type = "unknown";
    u64 vs_hash = 0;
    u64 fs_hash = 0;
    u64 cs_hash = 0;
    u32 write_count = 0;
};

struct WriterHistoryEntry {
    u64 submit_id;
    const char* writer_type;
    u64 vs_hash;
    u64 fs_hash;
    u64 cs_hash;
    u32 num_indices;
    u32 num_instances;
    u64 output_addr;
    bool blend_en;
    u32 write_mask;
    u32 rt_fmt;
    u32 depth_en;
    u32 depth_write;
    u32 color_export;
    u32 num_tex;
};

static constexpr u32 HISTORY_SIZE = 50;
static WriterHistoryEntry g_writer_history[HISTORY_SIZE];
static u32 g_writer_history_pos = 0;
static std::mutex g_history_mtx;
static std::map<u64, VkImageWriter> vk_image_writers; // gpu_addr -> last writer

void AddWriterHistory(u64 submit_id, const char* type, u64 vs, u64 fs, u64 cs,
                      u32 idx, u32 inst, u64 addr, bool blend, u32 wmask,
                      u32 rt_fmt, u32 depth_en, u32 depth_write, u32 color_export,
                      u32 num_tex) {
    auto& e = g_writer_history[g_writer_history_pos % HISTORY_SIZE];
    e.submit_id = submit_id;
    e.writer_type = type;
    e.vs_hash = vs & 0xFFFFFFFF;
    e.fs_hash = fs & 0xFFFFFFFF;
    e.cs_hash = cs & 0xFFFFFFFF;
    e.num_indices = idx;
    e.num_instances = inst;
    e.output_addr = addr;
    e.blend_en = blend;
    e.write_mask = wmask;
    e.rt_fmt = rt_fmt;
    e.depth_en = depth_en;
    e.depth_write = depth_write;
    e.color_export = color_export;
    e.num_tex = num_tex;
    g_writer_history_pos++;
}

void DumpWriterHistory() {
    u32 start = (g_writer_history_pos > HISTORY_SIZE) ? (g_writer_history_pos - HISTORY_SIZE) : 0;
    u32 end = g_writer_history_pos;
    LOG_INFO(Render_Vulkan, "KNACK_FRAME_WRITER_HISTORY_BEGIN total={}", g_writer_history_pos);
    for (u32 i = start; i < end; ++i) {
        const auto& e = g_writer_history[i % HISTORY_SIZE];
        LOG_INFO(Render_Vulkan,
                 "KNACK_FRAME_WRITER_HISTORY_ENTRY seq={} submit={} type={} "
                 "vs=0x{:08x} fs=0x{:08x} cs=0x{:08x} "
                 "idx={} inst={} addr=0x{:016x} blend={} wmask=0x{:x} "
                 "rt_fmt={} depth_en={} depth_write={} color_export=0x{:08x} num_tex={}",
                 i, e.submit_id, e.writer_type, (u32)e.vs_hash, (u32)e.fs_hash, (u32)e.cs_hash,
                 e.num_indices, e.num_instances, e.output_addr, e.blend_en, e.write_mask,
                 e.rt_fmt, e.depth_en, e.depth_write, e.color_export, e.num_tex);
    }
    LOG_INFO(Render_Vulkan, "KNACK_FRAME_WRITER_HISTORY_END");
}
} // namespace

void VkImageRecordColorWrite(u64 gpu_addr, u64 vs_hash, u64 fs_hash,
                             u32 wmask, bool blend, u32 rt_fmt, u32 depth_en,
                             u32 depth_write, u32 color_export, u32 num_tex) {
    auto& w = vk_image_writers[gpu_addr];
    w.writer_type = "color_attachment";
    w.vs_hash = vs_hash;
    w.fs_hash = fs_hash;
    w.write_count++;
    if (gpu_addr == 0x2a8ea0000) {
        AddWriterHistory(g_submit_id.load(), "color_attachment", vs_hash, fs_hash, 0,
                         6, 1, gpu_addr, blend, wmask, rt_fmt, depth_en, depth_write,
                         color_export, num_tex);
    }
    LOG_INFO(Render_Vulkan, "KNACK_VK_IMAGE_COLOR_WRITE addr=0x{:016x} vs=0x{:08x} fs=0x{:08x} count={}",
             gpu_addr, (u32)vs_hash, (u32)fs_hash, w.write_count);
}

void VkImageRecordComputeWrite(u64 gpu_addr, u64 cs_hash) {
    auto& w = vk_image_writers[gpu_addr];
    w.writer_type = "compute_storage";
    w.cs_hash = cs_hash;
    w.write_count++;
    if (gpu_addr == 0x2a8ea0000) {
        AddWriterHistory(g_submit_id.load(), "compute_storage", 0, 0, cs_hash,
                         0, 0, gpu_addr, false, 0, 0, 0, 0, 0, 0);
    }
    LOG_INFO(Render_Vulkan, "KNACK_VK_IMAGE_COMPUTE_WRITE addr=0x{:016x} cs=0x{:08x} count={}",
             gpu_addr, (u32)cs_hash, w.write_count);
}

void VkImageRecordFinalSample(u64 gpu_addr) {
    auto it = vk_image_writers.find(gpu_addr);
    if (it != vk_image_writers.end()) {
        LOG_INFO(Render_Vulkan,
                 "KNACK_FINAL_IMAGE_SAMPLE_WITH_LAST_WRITER addr=0x{:016x} last_writer={} "
                 "vs=0x{:08x} fs=0x{:08x} cs=0x{:08x} write_count={}",
                 gpu_addr, it->second.writer_type, (u32)it->second.vs_hash,
                 (u32)it->second.fs_hash, (u32)it->second.cs_hash, it->second.write_count);
    } else {
        LOG_INFO(Render_Vulkan, "KNACK_FINAL_IMAGE_UNKNOWN_WRITER addr=0x{:016x}", gpu_addr);
    }
    // Dump writer history for frame image
    DumpWriterHistory();
}

const char* VkImageGetLastWriter(u64 gpu_addr) {
    auto it = vk_image_writers.find(gpu_addr);
    return (it != vk_image_writers.end()) ? it->second.writer_type : "unknown";
}

// ─── Tornado Capture ───────────────────────────────────────────────

TornadoCapture& TornadoCapture::Instance() {
    static TornadoCapture tc;
    return tc;
}

void TornadoCapture::CheckTrigger() {
    if (active) return;
    static bool checked = false;
    if (!checked && std::filesystem::exists("KNACK_TORNADO_CAPTURE_NOW.txt")) {
        checked = true;
        std::filesystem::remove("KNACK_TORNADO_CAPTURE_NOW.txt");
        active = true;
        capture_frame = 0;
        std::filesystem::create_directories("dumps/knack_tornado_capture");
        csv.open("dumps/knack_tornado_capture/frames.csv");
        csv << "frame,draw,vs_hash,fs_hash,gs_hash,cs_hash,num_idx,num_inst,out_addr,"
               "out_fmt,wmask,blend,depth_en,depth_write,color_exp,shader_mask,target_mask\n";
        LOG_INFO(Render_Vulkan, "KNACK_TORNADO_CAPTURE_BEGIN frames={}", CAPTURE_MAX);
    }
}

bool TornadoCapture::IsCapturing() const { return active; }

void TornadoCapture::RecordDraw(u64 vs, u64 fs, u64 gs, u64 cs, u32 idx, u32 inst,
                                 u64 out_addr, u32 out_fmt, u32 wmask, bool blend,
                                 u32 depth_en, u32 depth_write, u32 color_exp,
                                 u32 shader_mask, u32 target_mask) {
    std::lock_guard lock(mtx);
    if (!active) return;
    if (capture_frame >= CAPTURE_MAX) {
        active = false;
        csv.close();
        LOG_INFO(Render_Vulkan, "KNACK_TORNADO_CAPTURE_END total_frames={}", capture_frame);
        return;
    }
    csv << capture_frame << "," << total_frames << ",0x" << fmt::format("{:08x}", (u32)vs)
        << ",0x" << fmt::format("{:08x}", (u32)fs) << ",0x" << fmt::format("{:08x}", (u32)gs)
        << ",0x" << fmt::format("{:08x}", (u32)cs) << "," << idx << "," << inst << ",0x"
        << fmt::format("{:016x}", out_addr) << "," << out_fmt << "," << wmask << "," << blend
        << "," << depth_en << "," << depth_write << ",0x" << fmt::format("{:08x}", color_exp)
        << ",0x" << fmt::format("{:02x}", shader_mask) << ",0x" << fmt::format("{:02x}", target_mask)
        << "\n";
    csv.flush();
}

void TornadoCapture::RecordImageSample(u32 slot, u64 addr, u32 fmt, u32 tile, u32 w, u32 h,
                                        bool same_output, const char* writer) {
    if (!active) return;
    LOG_INFO(Render_Vulkan,
             "KNACK_TORNADO_CAPTURE_IMAGE slot={} addr=0x{:016x} fmt={} tile={} {}x{} same_out={} writer={}",
             slot, addr, fmt, tile, w, h, same_output, writer);
}

void TornadoCapture::EndFrame() {
    if (!active) return;
    capture_frame++;
    if (capture_frame % 30 == 0) {
        LOG_INFO(Render_Vulkan, "KNACK_TORNADO_CAPTURE_FRAME frame={}", capture_frame);
    }
}

u64 TornadoCapture::HashGuest(u64 addr, u32 size) {
    const u8* p = reinterpret_cast<const u8*>(addr);
    if (!p) return 0;
    return XXH3_64bits(p, std::min(size, 64u * 1024u));
}

void TornadoCapture::DumpBmp(const std::string& path, u64 addr, u32 w, u32 h) {
    const u8* src = reinterpret_cast<const u8*>(addr);
    if (!src) return;
    const u32 dw = std::min(w, 320u), dh = std::min(h, 180u);
    std::ofstream bmp(path, std::ios::binary);
    const u32 row_size = (dw * 3 + 3) & ~3u;
    u8 hdr[54] = {};
    hdr[0]='B';hdr[1]='M';*(u32*)(hdr+2)=54+row_size*dh;*(u32*)(hdr+10)=54;
    *(u32*)(hdr+14)=40;*(s32*)(hdr+18)=dw;*(s32*)(hdr+22)=-(s32)dh;
    *(u16*)(hdr+26)=1;*(u16*)(hdr+28)=24;
    bmp.write((char*)hdr,54);
    std::vector<u8> row(row_size);
    for (u32 y=0;y<dh;++y){
        std::memset(row.data(),0,row_size);
        for(u32 x=0;x<dw;++x){
            u32 off=(y*2560+x)*4;
            row[x*3+0]=src[off+2];row[x*3+1]=src[off+1];row[x*3+2]=src[off+0];
        }
        bmp.write((char*)row.data(),row_size);
    }
    bmp.close();
}

void TornadoRecordDraw(u64 vs, u64 fs, u64 gs, u64 cs, u32 idx, u32 inst,
                       u64 out_addr, u32 out_fmt, u32 wmask, bool blend,
                       u32 depth_en, u32 depth_write, u32 color_exp,
                       u32 shader_mask, u32 target_mask) {
    TornadoCapture::Instance().RecordDraw(vs, fs, gs, cs, idx, inst, out_addr, out_fmt, wmask,
                                          blend, depth_en, depth_write, color_exp, shader_mask, target_mask);
}
void TornadoRecordSample(u32 slot, u64 addr, u32 fmt, u32 tile, u32 w, u32 h,
                         bool same_out, const char* writer) {
    TornadoCapture::Instance().RecordImageSample(slot, addr, fmt, tile, w, h, same_out, writer);
}
void TornadoEndFrame() { TornadoCapture::Instance().EndFrame(); }

} // namespace KnackDiag
