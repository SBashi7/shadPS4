// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <thread>
#include <boost/preprocessor/stringize.hpp>

#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/fence_detector.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/screenshot.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

// KNACK PM4 diagnostic rate limiting
static std::atomic<bool> knack_flip_signaled{false};
static std::atomic<u32> knack_pm4_type0_count{0};
static std::atomic<u32> knack_nextpacket_overflow_count{0};
static constexpr u32 KNACK_DIAG_MAX_FULL_DUMPS = 10;
static constexpr u32 KNACK_DIAG_SUMMARY_INTERVAL = 100;

// KNACK per-packet trace
static std::atomic<u32> knack_submit_index{0};

struct PacketTraceEntry {
    u32 packet_index = 0;
    size_t offset_dwords = 0;
    size_t remaining_dwords = 0;
    u32 raw_header = 0;
    u32 type = 0;
    u32 opcode = 0;
    u32 count_field = 0;
    u32 packet_total_dwords = 0;
};
static constexpr u32 TRACE_RING_SIZE = 5;
static PacketTraceEntry trace_ring[TRACE_RING_SIZE];
static u32 trace_ring_pos = 0;

static void PushTrace(const PacketTraceEntry& e) {
    trace_ring[trace_ring_pos % TRACE_RING_SIZE] = e;
    trace_ring_pos++;
}

static void DumpTraceRing() {
    const u32 count = std::min<u32>(trace_ring_pos, TRACE_RING_SIZE);
    const u32 start = (trace_ring_pos > TRACE_RING_SIZE) ? (trace_ring_pos - TRACE_RING_SIZE) : 0;
    for (u32 i = 0; i < count; i++) {
        const auto& e = trace_ring[(start + i) % TRACE_RING_SIZE];
        LOG_ERROR(Lib_GnmDriver,
                  "KNACK_PM4_TRACE_LAST_PACKET[{}] idx={} offset={} rem={} header=0x{:08x} "
                  "type={} opcode={} count={} pkt_dwords={}",
                  i, e.packet_index, e.offset_dwords, e.remaining_dwords, e.raw_header, e.type,
                  e.opcode, e.count_field, e.packet_total_dwords);
    }
}

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        const u32 count = knack_nextpacket_overflow_count.fetch_add(1);
        const bool full_dump = count < KNACK_DIAG_MAX_FULL_DUMPS;
        const bool summary = (count % KNACK_DIAG_SUMMARY_INTERVAL) == 0;

        if (full_dump) {
            // Dump current packet header and surrounding context
            const u32* data = span.data();
            const size_t remaining = span.size();

            LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_OVERFLOW #{}", count);
            LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_REQUESTED_DWORDS = {}", offset);
            LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_REMAINING_DWORDS = {}", remaining);
            LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_CURRENT_OFFSET = {} dwords from span start",
                      0);
            if (remaining > 0) {
                LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_CURRENT_HEADER = 0x{:08x}", data[0]);
            }
            // Dump last trace packets
            DumpTraceRing();
            // Dump remaining dwords (what's left in the span)
            {
                const size_t dump_n = std::min<size_t>(remaining, 128);
                for (size_t i = 0; i < dump_n; ++i) {
                    LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_DCB_TAIL[{}] = 0x{:08x}", i,
                              data[i]);
                }
            }
            LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_STOPPING_CURRENT_BUFFER");
        } else if (summary) {
            LOG_ERROR(Lib_GnmDriver, "KNACK_NEXTPACKET_OVERFLOW summary: {} total occurrences",
                      count);
        }
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();
        if (VideoCore::ConsumeScreenshotRequest()) {
            if (rasterizer) {
                VideoCore::CaptureScreenshot(*rasterizer, {});
            }
        }
        curr_qid = -1;

        while (num_submits || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();

                --num_submits;
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
            }
        }

        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }

        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }
    const bool host_markers_enabled = rasterizer && Config::getVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && Config::getVkGuestMarkersEnabled();

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    const u32* const initial_dcb_data = dcb.data();
    const size_t initial_dcb_size = dcb.size();
    const u32 this_submit = knack_submit_index.fetch_add(1);
    u32 packet_index = 0;
    const bool trace_enabled = this_submit < 5;
    constexpr u32 TRACE_MAX_PACKETS = 1000;

    LOG_ERROR(Lib_GnmDriver, "KNACK_PROCESSGRAPHICS_ENTER submit={} dcb_size={}", this_submit,
              initial_dcb_size);
    LOG_ERROR(Lib_GnmDriver, "KNACK_GPU_TASK_ACTIVE submit={}", this_submit);

    // KNACK: fallback PatchedFlip scan at ProcessGraphics entry
    if (initial_dcb_size >= 64) {
        const u32* scan = initial_dcb_data;
        bool found = false;
        size_t found_off = 0;
        for (size_t i = 0; i + 1 < initial_dcb_size; ++i) {
            if (scan[i] == 0xc0391000 && scan[i + 1] == 0x68750776) {
                found = true;
                found_off = i;
                break;
            }
        }
        if (found) {
            // Validate: PatchedFlip Nop must be at the expected tail offset (size - 59)
            const size_t expected_nop_off = initial_dcb_size - 59;
            if (found_off != expected_nop_off && found_off != initial_dcb_size - 59) {
                LOG_ERROR(Lib_GnmDriver,
                          "KNACK_PATCHEDFLIP_WRONG_OFFSET submit={} found={} expected={} "
                          "dcb_size={} — skipping label fallback",
                          this_submit, found_off, expected_nop_off, initial_dcb_size);
                // Still signal GfxFlip (the marker IS present, just at wrong offset)
                found = false; // prevent label fallback below
            }

            LOG_ERROR(Lib_GnmDriver,
                      "KNACK_PATCHEDFLIP_SCAN_FOUND submit={} offset={} header=0x{:08x} "
                      "payload=0x{:08x}",
                      this_submit, found_off, scan[found_off], scan[found_off + 1]);
            LOG_ERROR(Lib_GnmDriver, "KNACK_PATCHEDFLIP_FALLBACK_SIGNAL submit={}", this_submit);
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);

            // Also directly reset the VO label to break circular deadlock
            // The flip WriteData is 5 dwords before the Nop (count=3, total 5 dwords)
            if (found && found_off >= 5) {
                const u32 wd_header = scan[found_off - 5];
                const u32 wd_count = (wd_header >> 16) & 0x3FFF;
                const u32 wd_opcode = (wd_header >> 8) & 0xFF;
                if (wd_opcode == 0x37 && wd_count >= 3) {
                    const u32 addr_lo = scan[found_off - 3];
                    const u32 addr_hi = scan[found_off - 2];
                    const u64 label_addr64 =
                        (static_cast<u64>(addr_hi) << 32) | static_cast<u64>(addr_lo);
                    const u32 data_val = scan[found_off - 1];
                    LOG_ERROR(Lib_GnmDriver,
                              "KNACK_VO_LABEL_FALLBACK_ENTER submit={} label_addr={:p} old_val={}",
                              this_submit, fmt::ptr(reinterpret_cast<void*>(label_addr64)),
                              data_val);
                    if (data_val != 0) {
                        u64* label_ptr = reinterpret_cast<u64*>(label_addr64);
                        LOG_ERROR(Lib_GnmDriver,
                                  "KNACK_VO_LABEL_FALLBACK_WRITE_ZERO label_addr={:p} old_val={} "
                                  "new_val=0",
                                  fmt::ptr(reinterpret_cast<void*>(label_addr64)), *label_ptr);
                        *label_ptr = 0;
                        if (vo_port && vo_port->IsVoLabel(label_ptr)) {
                            vo_port->SignalVoLabel();
                        }
                        LOG_ERROR(Lib_GnmDriver,
                                  "KNACK_VO_LABEL_FALLBACK_NOTIFY submit={} label_addr={:p}",
                                  this_submit, fmt::ptr(reinterpret_cast<void*>(label_addr64)));
                    } else {
                        LOG_ERROR(Lib_GnmDriver,
                                  "KNACK_VO_LABEL_FALLBACK_SKIP submit={} reason=already_zero",
                                  this_submit);
                    }
                } else {
                    LOG_ERROR(Lib_GnmDriver,
                              "KNACK_VO_LABEL_FALLBACK_SKIP submit={} "
                              "reason=no_writedata_at_Nop-5 opcode={} count={}",
                              this_submit, wd_opcode, wd_count);
                }
            }
        } else {
            LOG_ERROR(Lib_GnmDriver, "KNACK_PATCHEDFLIP_SCAN_NOT_FOUND submit={}", this_submit);
        }
    }

    // Dump first 128 dwords for trace-enabled submits
    if (trace_enabled) {
        const size_t head_n = std::min<size_t>(dcb.size(), 128);
        for (size_t i = 0; i < head_n; ++i) {
            LOG_ERROR(Lib_GnmDriver, "KNACK_DCB_HEAD submit={} [{}] = 0x{:08x}", this_submit, i,
                      dcb[i]);
        }
        // Log expected flip patch positions for this DCB
        if (dcb.size() >= 64) {
            const size_t flip_write = dcb.size() - 64;
            const size_t flip_nop = flip_write + 5;
            const size_t flip_payload = flip_nop + 1;
            LOG_ERROR(Lib_GnmDriver,
                      "KNACK_PATCHEDFLIP_EXPECTED submit={} flip_write_offset={} nop_offset={} "
                      "payload_offset={} expected_nop=0xC0391000 expected_payload=0x68750776",
                      this_submit, flip_write, flip_nop, flip_payload);
        }
    }

    while (!dcb.empty()) {
        if (this_submit == 3) {
            LOG_ERROR(Lib_GnmDriver, "KNACK_PG_BEFORE_PROCESSCOMMANDS submit=3");
        }
        ProcessCommands();

        if (this_submit == 3) {
            LOG_ERROR(Lib_GnmDriver, "KNACK_PG_AFTER_PROCESSCOMMANDS submit=3");
        }

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        // Per-packet logging for submit #3
        if (this_submit == 3 && packet_index < TRACE_MAX_PACKETS) {
            const size_t off =
                reinterpret_cast<const u32*>(header) - reinterpret_cast<const u32*>(base_addr);
            u32 t3_opcode = 0;
            u32 t3_count = 0;
            u32 t3_total = 0;
            if (type == 3) {
                t3_opcode = static_cast<u32>(header->type3.opcode.Value());
                t3_count = header->type3.NumWords();
                t3_total = t3_count + 1;
            }
            LOG_ERROR(Lib_GnmDriver,
                      "KNACK_PG_PACKET_BEGIN submit=3 pkt={} offset={} rem={} header=0x{:08x} "
                      "type={} opcode={} count={} total_dw={}",
                      packet_index, off, dcb.size(), header->raw, type, t3_opcode, t3_count,
                      t3_total);
        }

        // Per-packet trace
        if (trace_enabled && packet_index < TRACE_MAX_PACKETS) {
            PacketTraceEntry entry{};
            entry.packet_index = packet_index;
            entry.offset_dwords =
                reinterpret_cast<const u32*>(header) - reinterpret_cast<const u32*>(base_addr);
            entry.remaining_dwords = dcb.size();
            entry.raw_header = header->raw;
            entry.type = type;
            entry.count_field = header->type3.count.Value();
            if (type == 3) {
                entry.opcode = static_cast<u32>(header->type3.opcode.Value());
                entry.packet_total_dwords = header->type3.NumWords() + 1;
            } else if (type == 0) {
                entry.packet_total_dwords = header->type0.NumWords() + 1;
            } else {
                entry.packet_total_dwords = 1;
            }

            // Detect advance mismatches
            static size_t prev_expected_next = 0;
            static u32 prev_opcode = 0;
            static u32 prev_count_field = 0;
            static u32 prev_pkt_index = 0;
            const size_t expected_start = prev_expected_next;
            const size_t actual_start = entry.offset_dwords;
            if (packet_index > 0 && expected_start != actual_start) {
                LOG_ERROR(Lib_GnmDriver,
                          "KNACK_PM4_ADVANCE_MISMATCH submit={} prev_pkt={} prev_opcode={} "
                          "prev_count={} expected_next={} actual_start={} diff={}",
                          this_submit, prev_pkt_index, prev_opcode, prev_count_field,
                          expected_start, actual_start, (s64)actual_start - (s64)expected_start);
            }
            prev_expected_next = entry.offset_dwords + entry.packet_total_dwords;
            prev_opcode = entry.opcode;
            prev_count_field = entry.count_field;
            prev_pkt_index = entry.packet_index;

            // Tail range trace: log every packet when remaining <= 200 dwords
            if (entry.remaining_dwords <= 200) {
                const size_t abs_off = entry.offset_dwords;
                const size_t end_off = abs_off + entry.packet_total_dwords;
                const char* opcode_name = (type == 3) ? "type3" : (type == 0) ? "type0" : "other";
                LOG_ERROR(Lib_GnmDriver,
                          "KNACK_PM4_RANGE submit={} pkt={} start={} end={} rem={} "
                          "header=0x{:08x} type={} opcode={} count={} total_dw={}",
                          this_submit, entry.packet_index, abs_off, end_off, entry.remaining_dwords,
                          entry.raw_header, entry.type, entry.opcode, entry.count_field,
                          entry.packet_total_dwords);
            }

            PushTrace(entry);
            packet_index++;
        }

        switch (type) {
        default: {
            // KNACK: stop buffer after too many unknown types to prevent exit freeze
            // Use a per-call static that resets on ProcessGraphics exit
            static u32 unknown_skip_buf = 0;
            if (packet_index == 0)
                unknown_skip_buf = 0;
            unknown_skip_buf++;
            if (unknown_skip_buf > 5000) {
                LOG_ERROR(Lib_GnmDriver, "KNACK_UNKNOWN_TYPE_STOP buf_skips={}", unknown_skip_buf);
                unknown_skip_buf = 0;
                dcb = {};
                break;
            }
            dcb = NextPacket(dcb, 1);
            continue;
        }
        case 0: {
            const u32 count = knack_pm4_type0_count.fetch_add(1);
            const bool full_dump = count < KNACK_DIAG_MAX_FULL_DUMPS;
            const bool summary = (count % KNACK_DIAG_SUMMARY_INTERVAL) == 0;

            if (full_dump || summary) {
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_HIT #{} {}", count,
                          full_dump ? "(full dump)" : "(summary)");
            }

            if (full_dump) {
                const u32 raw = header->raw;
                const u32 num_words = header->type0.NumWords();
                const u32 skip = num_words + 1;
                const size_t current_offset = reinterpret_cast<const u32*>(header) - dcb.data();
                const size_t remaining = dcb.size();

                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_RAW_HEADER = 0x{:08x}", raw);
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_COUNT bits = {}",
                          header->type0.count.Value());
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_NUM_WORDS = {}", num_words);
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_SKIP_DWORDS = {}", skip);
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_OFFSET_DWORDS = {}", current_offset);
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_REMAINING_DWORDS = {}", remaining);
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_THREAD_ID = {}",
                          std::hash<std::thread::id>{}(std::this_thread::get_id()));

                // Dump previous 8 dwords (if available)
                {
                    const u32* base = dcb.data();
                    const size_t start = (current_offset >= 8) ? (current_offset - 8) : 0;
                    const size_t n = current_offset - start;
                    const size_t max_show = 8;
                    for (size_t i = 0; i < n && i < max_show; ++i) {
                        LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_PREV_DWORDS[{}] = 0x{:08x}", i,
                                  base[start + i]);
                    }
                }

                // Dump header + next 16 dwords
                {
                    const u32* base = dcb.data();
                    const size_t max_dump = std::min<size_t>(remaining, 17); // header + 16 next
                    for (size_t i = 0; i < max_dump; ++i) {
                        LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_NEXT_DWORDS[{}] = 0x{:08x}", i,
                                  base[current_offset + i]);
                    }
                }

                // Check if header appears to be zero/uninitialized
                if (raw == 0) {
                    LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_ZERO_HEADER_DETECTED");
                }
            } else if (summary) {
                LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_TYPE0_HIT summary: {} total occurrences",
                          count);
            }

            dcb = NextPacket(dcb, header->type0.NumWords() + 1);
            continue;
        }
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const u32 pkt_total = count + 1;
            // KNACK: guard against garbage tail parsed as Nop (false PatchedFlip)
            if (pkt_total > dcb.size()) {
                const u32 opcode_raw = static_cast<u32>(header->type3.opcode.Value());
                LOG_ERROR(Lib_GnmDriver,
                          "KNACK_PM4_OVERFLOW_GUARD pkt_total={} remaining={} opcode={} "
                          "header=0x{:08x}",
                          pkt_total, dcb.size(), opcode_raw, header->raw);
                if (opcode_raw == 0x10) {
                    const u32* tail = dcb.data();
                    size_t zero_count = 0;
                    const size_t check_n = std::min<size_t>(dcb.size(), 44);
                    for (size_t i = 2; i < check_n; ++i) {
                        if (tail[i] == 0)
                            zero_count++;
                    }
                    if (zero_count >= check_n / 2) {
                        LOG_ERROR(Lib_GnmDriver, "KNACK_PM4_OVERFLOW_GUARD_SKIP_TAIL zeros={}/{}",
                                  zero_count, check_n - 2);
                        LOG_ERROR(
                            Lib_GnmDriver,
                            "KNACK_PROCESSGRAPHICS_EARLY_STOP submit={} reason=overflow_guard "
                            "offset={} rem={} header=0x{:08x}",
                            this_submit,
                            reinterpret_cast<const u32*>(header) -
                                reinterpret_cast<const u32*>(base_addr),
                            dcb.size(), header->raw);
                        dcb = {};
                        break;
                    }
                }
            }
            const PM4ItOpcode opcode = header->type3.opcode;
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                const u32 nop_count = nop->header.count.Value();
                LOG_ERROR(Lib_GnmDriver, "KNACK_NOP_SEEN count={} payload0=0x{:08x}", nop_count,
                          nop_count > 0 ? nop->data_block[0] : 0);
                if (nop_count == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    LOG_ERROR(Lib_GnmDriver, "KNACK_PATCHEDFLIP_SIGNAL");
                    knack_flip_signaled.store(true);
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_DEBUG(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndex2", cmd_address));
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexOffset2", cmd_address));
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirect", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirect", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectCountMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchDirect", cmd_address));
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchIndirect", cmd_address));
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size, true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size, false);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                LOG_ERROR(Lib_GnmDriver, "KNACK_EVENTWRITE_CALLED");
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                LOG_ERROR(Lib_GnmDriver, "KNACK_EVENTWRITEEOS_CALLED");
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (rasterizer) {
                    rasterizer->CommitPendingGpuRanges();
                }
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                        memcpy(address, &data, num_bytes);
                    }
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                LOG_ERROR(Lib_GnmDriver, "KNACK_EVENTWRITEEOP_CALLED");
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (rasterizer) {
                    rasterizer->CommitPendingGpuRanges();
                }
                event_eop->SignalFence(
                    [](void* address, u64 data, u32 num_bytes) {
                        auto* memory = Core::Memory::Instance();
                        if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                            memcpy(address, &data, num_bytes);
                        }
                    },
                    [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                    u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                LOG_ERROR(Lib_GnmDriver, "KNACK_WRITEDATA_CALLED");
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (data_size <= sizeof(u64) && rasterizer) {
                    rasterizer->CommitPendingGpuRanges();
                }
                if (!write_data->wr_one_addr.Value()) {
                    std::memcpy(address, write_data->data, data_size);
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_DEBUG(Render,
                          "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                          "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                          u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                          copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                          u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                LOG_ERROR(Lib_GnmDriver, "KNACK_ACQUIREMEM_CALLED");
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                rasterizer->CommitPendingGpuRanges();
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                LOG_ERROR(Lib_GnmDriver, "KNACK_WAITREGMEM_CALLED");
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (this_submit == 3) {
                    LOG_ERROR(Lib_GnmDriver,
                              "KNACK_WAITREGMEM_ENTER submit=3 offset={} addr={:p} func={} "
                              "ref={} mask={} poll_interval={} current_value={}",
                              reinterpret_cast<const u32*>(header) -
                                  reinterpret_cast<const u32*>(base_addr),
                              fmt::ptr(wait_addr), u32(wait_reg_mem->function.Value()),
                              wait_reg_mem->ref, wait_reg_mem->mask, wait_reg_mem->poll_interval,
                              *wait_addr);
                }
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    if (this_submit == 3) {
                        LOG_ERROR(Lib_GnmDriver, "KNACK_WAITREGMEM_VO_LABEL_WAIT submit=3");
                    }
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    YIELD_GFX();
                    if (this_submit == 3) {
                        static u32 yield_count_3 = 0;
                        if (++yield_count_3 % 1000 == 0) {
                            LOG_ERROR(Lib_GnmDriver,
                                      "KNACK_WAITREGMEM_STILL_WAITING submit=3 yields={}",
                                      yield_count_3);
                        }
                    }
                }
                if (this_submit == 3) {
                    LOG_ERROR(Lib_GnmDriver, "KNACK_WAITREGMEM_EXIT submit=3");
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                if (rasterizer) {
                    rasterizer->CpSync();
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_DEBUG(Render_Vulkan,
                          "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                          "source_select = {}, buffer_select = {}",
                          strmout->update_memory.Value(),
                          magic_enum::enum_name(strmout->source_select.Value()),
                          strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_DEBUG(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_DEBUG(Render, "IT_COND_EXEC used a reserved command");
                }
                const bool skip = (*cond_exec->Address() == false);
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }

            default:
                // UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                //                 static_cast<u32>(opcode), count);
                break;
            }

            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    LOG_ERROR(Lib_GnmDriver, "KNACK_PROCESSGRAPHICS_EXIT submit={}", this_submit);
    LOG_ERROR(Lib_GnmDriver, "KNACK_GPU_TASK_FINISHED submit={}", this_submit);

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];
    const bool host_markers_enabled = rasterizer && Config::getVkHostMarkersEnabled();

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();

    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                }
                rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            auto& buffer_cache = rasterizer->GetBufferCache();
            auto& gpu_modified_ranges_pending = buffer_cache.GetPendingGpuModifiedRanges();
            const VAddr rewind_addr = reinterpret_cast<VAddr>(acb.data());
            bool must_flush = false;
            gpu_modified_ranges_pending.ForEachInRange(
                rewind_addr, acb.size_bytes(),
                [&indirect_patches, &must_flush, rewind_header = header](VAddr begin, VAddr end) {
                    const u32 range_size = end - begin;
                    if (range_size != sizeof(PM4CmdDispatchIndirect::GroupDimensions)) {
                        must_flush = true;
                        return;
                    }
                    const PM4Header* header =
                        reinterpret_cast<const PM4Header*>(begin - sizeof(PM4Header));
                    if (header->type != 3 || header->type3.opcode != PM4ItOpcode::DispatchDirect) {
                        must_flush = true;
                        return;
                    }
                    // FIX: Use emplace_back and provide the 'begin' address as the second argument
                    indirect_patches.emplace_back(header, begin);
                });

            if (must_flush) {
                buffer_cache.CommitPendingGpuRanges();
                rasterizer->CommitPendingGpuRanges();

            } else {
                // All GPU modified regions in the command list are patched with indirect
                // dispatches. There is no needed to flush GPU data to CPU so avoiding
                // read-protecting pages.
                gpu_modified_ranges_pending.Subtract(rewind_addr, acb.size_bytes());
            }

            //  ASSERT_MSG(rewind->Valid(), "Rewind valid bit must be set");
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_DEBUG(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                      set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size, true);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                    rasterizer->DispatchDirect();
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchDirect();
                }
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            cs_program.dispatch_initiator = dispatch_indirect->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                    rasterizer->DispatchIndirect(ib_address, 0, size, true);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchIndirect(ib_address, 0, size, false);
                }
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (data_size <= sizeof(u64) && rasterizer) {
                rasterizer->CommitPendingGpuRanges();
            }
            if (!write_data->wr_one_addr.Value()) {
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            while (!wait_reg_mem->Test(regs.reg_array)) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (rasterizer) {
                rasterizer->CommitPendingGpuRanges();
            }
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    static std::atomic<u32> submit_count{0};
    const u32 n = submit_count.fetch_add(1);
    const bool copy_enabled = Config::copyGPUCmdBuffers();

    if (n < 5) {
        LOG_ERROR(Lib_GnmDriver, "KNACK_COPY_GPU_BUFFERS_RUNTIME_{}",
                  copy_enabled ? "TRUE" : "FALSE");
        LOG_ERROR(Lib_GnmDriver, "KNACK_SUBMITGFX_ENTER submit_id={} dcb_size={} ccb_size={}", n,
                  dcb.size(), ccb.size());
    }

    if (Config::copyGPUCmdBuffers()) {
        if (n < 5) {
            LOG_ERROR(Lib_GnmDriver,
                      "KNACK_COPY_CMD_BUFFERS_CALLED #{} dcb_dwords={} ccb_dwords={}", n,
                      dcb.size(), ccb.size());
        }
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
        if (n < 5) {
            LOG_ERROR(Lib_GnmDriver,
                      "KNACK_COPY_CMD_BUFFERS_DONE #{} copied_dcb_size={} copied_ccb_size={}", n,
                      dcb.size(), ccb.size());
            // Log tail area where flip patch Nop should be (size-64 offset)
            if (dcb.size() >= 64) {
                const size_t flip_offset = dcb.size() - 64;
                LOG_ERROR(Lib_GnmDriver,
                          "KNACK_COPY_DCB_FLIP_TAIL #{} dcb[{}]=0x{:08x} dcb[{}]=0x{:08x} "
                          "dcb[{}]=0x{:08x}",
                          n, flip_offset, dcb[flip_offset], flip_offset + 5, dcb[flip_offset + 5],
                          flip_offset + 6, dcb[flip_offset + 6]);
            }
        }
    }

    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}

} // namespace AmdGpu