/*
OBS macOS Background Removal
Copyright (C) 2023-2024 Sebastian Beckmann

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <plugin-support.h>
#include <util/threading.h>
#include <Vision/Vision.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

/* Frames are kept on the GPU until their mask is ready, so that a mask is only ever composited onto the frame it was
 * computed from. Worst case occupancy is two frames in flight plus the one being displayed, plus one to render into. */
#define RING_SIZE 4
#define MAX_IN_FLIGHT_SYNCED 2
#define MAX_IN_FLIGHT_UNSYNCED 1

/* Vision resamples its input to a fixed model resolution anyway (masks are 2016x1512 in Accurate mode, 512x384 in
 * Balanced and 256x192 in Fast), so reading back more pixels than this from the GPU only costs time. */
#define ANALYSIS_MAX_DIMENSION_ACCURATE 1920
#define ANALYSIS_MAX_DIMENSION_OTHER 960

enum slot_state {
    SLOT_FREE,
    SLOT_IN_FLIGHT,
    SLOT_READY,
};

struct frame_slot {
    gs_texrender_t *texrender;
    uint64_t seq;
    enum slot_state state;
};

struct vision_data {
    obs_source_t *context;

    /* Graphics thread only */
    struct frame_slot slots[RING_SIZE];
    struct frame_slot *latest; /* Frame rendered during the current tick */
    struct frame_slot *ready;  /* Frame the current mask texture belongs to */
    gs_texrender_t *analysis;
    gs_stagesurf_t *stagesurf;
    gs_texture_t *mask_texture;
    bool have_mask;
    bool mask_logged;

    gs_effect_t *effect;
    gs_eparam_t *src_param;
    gs_eparam_t *mask_param;
    gs_eparam_t *mask_low_param;
    gs_eparam_t *mask_high_param;
    gs_eparam_t *mask_texel_param;
    gs_eparam_t *range_scale_param;

    uint64_t tick;
    uint64_t render_tick;
    uint64_t next_seq;
    uint64_t min_valid_seq;
    uint32_t in_flight;

    CVPixelBufferPoolRef pool;
    uint32_t pool_width;
    uint32_t pool_height;

    /* Settings */
    float threshold;
    float softness;
    float refinement;
    bool sync;
    VNGeneratePersonSegmentationRequestQualityLevel qualityLevel;

    /* Mask queue only */
    VNGeneratePersonSegmentationRequest *request;
    VNSequenceRequestHandler *handler;
    bool error_logged;

    dispatch_queue_t mask_queue;

    /* Shared, guarded by result_mutex */
    pthread_mutex_t result_mutex;
    CVPixelBufferRef result_mask;
    uint64_t result_seq;
    uint32_t results_completed;
};

static const char *vision_get_name(void *)
{
    return obs_module_text("Name");
}

static void vision_tick(void *filter_ptr, float)
{
    struct vision_data *filter = filter_ptr;
    filter->tick++;
}

static void upload_mask(struct vision_data *filter, CVPixelBufferRef mask)
{
    CVPixelBufferLockBaseAddress(mask, kCVPixelBufferLock_ReadOnly);

    const uint32_t width = (uint32_t) CVPixelBufferGetWidth(mask);
    const uint32_t height = (uint32_t) CVPixelBufferGetHeight(mask);
    if (filter->mask_texture && (width != gs_texture_get_width(filter->mask_texture) ||
                                 height != gs_texture_get_height(filter->mask_texture))) {
        gs_texture_destroy(filter->mask_texture);
        filter->mask_texture = NULL;
    }
    if (!filter->mask_texture) {
        filter->mask_texture = gs_texture_create(width, height, GS_A8, 1, NULL, GS_DYNAMIC);
    }
    if (filter->mask_texture) {
        gs_texture_set_image(filter->mask_texture, CVPixelBufferGetBaseAddress(mask),
                             (uint32_t) CVPixelBufferGetBytesPerRow(mask), false);
        filter->have_mask = true;
        if (!filter->mask_logged) {
            obs_log(LOG_INFO, "First mask received (%ux%u)", width, height);
            filter->mask_logged = true;
        }
    }

    CVPixelBufferUnlockBaseAddress(mask, kCVPixelBufferLock_ReadOnly);
}

/* Picks up whatever the mask queue has finished since the last tick. Results arrive in submission order, so only the
 * newest one matters; frames that were waiting on older results are released. */
static void collect_results(struct vision_data *filter)
{
    pthread_mutex_lock(&filter->result_mutex);
    CVPixelBufferRef mask = filter->result_mask;
    const uint64_t seq = filter->result_seq;
    const uint32_t completed = filter->results_completed;
    filter->result_mask = NULL;
    filter->results_completed = 0;
    pthread_mutex_unlock(&filter->result_mutex);

    if (!completed)
        return;

    filter->in_flight -= completed;

    for (size_t i = 0; i < RING_SIZE; i++) {
        struct frame_slot *slot = &filter->slots[i];
        if (slot->state != SLOT_IN_FLIGHT || slot->seq > seq)
            continue;

        if (slot->seq == seq && mask && seq >= filter->min_valid_seq) {
            upload_mask(filter, mask);
            if (filter->ready)
                filter->ready->state = SLOT_FREE;
            slot->state = SLOT_READY;
            filter->ready = slot;
        } else {
            slot->state = SLOT_FREE;
        }
    }

    if (mask)
        CVPixelBufferRelease(mask);
}

/* Called when rendering resumes after the source was not shown for a while. Everything we hold is stale by then, so
 * show nothing until a fresh mask arrives rather than an old frame or a mask that no longer fits. */
static void invalidate(struct vision_data *filter)
{
    filter->min_valid_seq = filter->next_seq;
    filter->have_mask = false;
    if (filter->ready) {
        filter->ready->state = SLOT_FREE;
        filter->ready = NULL;
    }
}

static struct frame_slot *get_free_slot(struct vision_data *filter)
{
    for (size_t i = 0; i < RING_SIZE; i++) {
        if (filter->slots[i].state == SLOT_FREE)
            return &filter->slots[i];
    }
    return NULL;
}

static bool render_target(struct vision_data *filter, gs_texrender_t *texrender, uint32_t width, uint32_t height)
{
    obs_source_t *target = obs_filter_get_target(filter->context);
    const obs_source_t *parent = obs_filter_get_parent(filter->context);

    const uint32_t target_flags = obs_source_get_output_flags(target);
    const bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
    const bool async_source = (target_flags & OBS_SOURCE_ASYNC) != 0;

    bool success = false;

    gs_texrender_reset(texrender);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    if (gs_texrender_begin(texrender, width, height)) {
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0, 0);
        gs_ortho(0, width, 0, height, -100, 100);
        if (target == parent && !custom_draw && !async_source) {
            obs_source_default_render(target);
        } else {
            obs_source_video_render(target);
        }
        gs_texrender_end(texrender);
        success = true;
    }
    gs_blend_state_pop();

    return success;
}

/* Scales the frame down to what Vision can make use of. Returns the texture to read back. */
static gs_texture_t *prepare_analysis_texture(struct vision_data *filter, gs_texture_t *source, uint32_t width,
                                              uint32_t height, uint32_t *out_width, uint32_t *out_height)
{
    const uint32_t limit = filter->qualityLevel == VNGeneratePersonSegmentationRequestQualityLevelAccurate
                               ? ANALYSIS_MAX_DIMENSION_ACCURATE
                               : ANALYSIS_MAX_DIMENSION_OTHER;
    const uint32_t longest = width > height ? width : height;

    *out_width = width;
    *out_height = height;
    if (longest <= limit)
        return source;

    const uint32_t scaled_width = (uint32_t) ((uint64_t) width * limit / longest);
    const uint32_t scaled_height = (uint32_t) ((uint64_t) height * limit / longest);
    if (!scaled_width || !scaled_height)
        return source;

    gs_texture_t *result = source;

    gs_texrender_reset(filter->analysis);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    if (gs_texrender_begin(filter->analysis, scaled_width, scaled_height)) {
        const bool previous_srgb = gs_framebuffer_srgb_enabled();
        gs_enable_framebuffer_srgb(false);

        gs_ortho(0, scaled_width, 0, scaled_height, -100, 100);
        gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture(gs_effect_get_param_by_name(default_effect, "image"), source);
        while (gs_effect_loop(default_effect, "Draw")) {
            gs_draw_sprite(source, 0, scaled_width, scaled_height);
        }

        gs_enable_framebuffer_srgb(previous_srgb);
        gs_texrender_end(filter->analysis);

        gs_texture_t *scaled = gs_texrender_get_texture(filter->analysis);
        if (scaled) {
            result = scaled;
            *out_width = scaled_width;
            *out_height = scaled_height;
        }
    }
    gs_blend_state_pop();

    return result;
}

/* Copies the frame off the GPU into a buffer that Vision owns for as long as it needs it. */
static CVPixelBufferRef copy_to_pixel_buffer(struct vision_data *filter, gs_texture_t *texture, uint32_t width,
                                             uint32_t height)
{
    if (filter->stagesurf && (width != gs_stagesurface_get_width(filter->stagesurf) ||
                              height != gs_stagesurface_get_height(filter->stagesurf))) {
        gs_stagesurface_destroy(filter->stagesurf);
        filter->stagesurf = NULL;
    }
    if (!filter->stagesurf) {
        filter->stagesurf = gs_stagesurface_create(width, height, GS_BGRA);
        if (!filter->stagesurf)
            return NULL;
    }

    if (filter->pool && (width != filter->pool_width || height != filter->pool_height)) {
        CVPixelBufferPoolRelease(filter->pool);
        filter->pool = NULL;
    }
    if (!filter->pool) {
        NSDictionary *attributes = @{
            (id) kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (id) kCVPixelBufferWidthKey: @(width),
            (id) kCVPixelBufferHeightKey: @(height),
            (id) kCVPixelBufferIOSurfacePropertiesKey: @ {},
        };
        if (CVPixelBufferPoolCreate(kCFAllocatorDefault, NULL, (CFDictionaryRef) attributes, &filter->pool) !=
            kCVReturnSuccess) {
            filter->pool = NULL;
            return NULL;
        }
        filter->pool_width = width;
        filter->pool_height = height;
    }

    CVPixelBufferRef pixel_buffer = NULL;
    if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, filter->pool, &pixel_buffer) != kCVReturnSuccess)
        return NULL;

    gs_stage_texture(filter->stagesurf, texture);

    uint8_t *data;
    uint32_t linesize;
    if (!gs_stagesurface_map(filter->stagesurf, &data, &linesize)) {
        CVPixelBufferRelease(pixel_buffer);
        return NULL;
    }

    CVPixelBufferLockBaseAddress(pixel_buffer, 0);
    uint8_t *destination = CVPixelBufferGetBaseAddress(pixel_buffer);
    const size_t destination_linesize = CVPixelBufferGetBytesPerRow(pixel_buffer);
    const size_t row_bytes = (size_t) width * 4;
    if (destination_linesize == linesize) {
        memcpy(destination, data, (size_t) linesize * (height - 1) + row_bytes);
    } else {
        for (uint32_t y = 0; y < height; y++) {
            memcpy(destination + destination_linesize * y, data + (size_t) linesize * y, row_bytes);
        }
    }
    CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);

    gs_stagesurface_unmap(filter->stagesurf);

    return pixel_buffer;
}

static bool submit_frame(struct vision_data *filter, struct frame_slot *slot, uint32_t width, uint32_t height)
{
    gs_texture_t *source = gs_texrender_get_texture(slot->texrender);
    if (!source)
        return false;

    uint32_t analysis_width, analysis_height;
    gs_texture_t *analysis = prepare_analysis_texture(filter, source, width, height, &analysis_width, &analysis_height);

    CVPixelBufferRef input = copy_to_pixel_buffer(filter, analysis, analysis_width, analysis_height);
    if (!input)
        return false;

    const uint64_t seq = filter->next_seq++;
    const VNGeneratePersonSegmentationRequestQualityLevel quality = filter->qualityLevel;

    slot->seq = seq;
    slot->state = SLOT_IN_FLIGHT;
    filter->in_flight++;

    dispatch_async(filter->mask_queue, ^{
        CVPixelBufferRef mask = NULL;

        @autoreleasepool {
            filter->request.qualityLevel = quality;

            NSError *error = nil;
            if ([filter->handler performRequests:@[filter->request] onCVPixelBuffer:input error:&error]) {
                mask = filter->request.results.firstObject.pixelBuffer;
                if (mask)
                    CVPixelBufferRetain(mask);
            } else if (!filter->error_logged) {
                obs_log(LOG_ERROR, "Person segmentation failed: %s", error.localizedDescription.UTF8String);
                filter->error_logged = true;
            }
        }

        CVPixelBufferRelease(input);

        pthread_mutex_lock(&filter->result_mutex);
        if (filter->result_mask)
            CVPixelBufferRelease(filter->result_mask);
        filter->result_mask = mask;
        filter->result_seq = seq;
        filter->results_completed++;
        pthread_mutex_unlock(&filter->result_mutex);
    });

    return true;
}

/* Runs once per tick, no matter how many times the filter is drawn. */
static void process_frame(struct vision_data *filter, uint32_t width, uint32_t height, bool resumed)
{
    if (resumed)
        invalidate(filter);

    collect_results(filter);

    filter->latest = NULL;
    struct frame_slot *slot = get_free_slot(filter);
    if (!slot || !render_target(filter, slot->texrender, width, height))
        return;
    filter->latest = slot;

    /* If Vision can't keep up, skip this frame instead of letting a backlog (and with it the delay) build up. */
    const uint32_t max_in_flight = filter->sync ? MAX_IN_FLIGHT_SYNCED : MAX_IN_FLIGHT_UNSYNCED;
    if (filter->in_flight < max_in_flight)
        submit_frame(filter, slot, width, height);
}

static void vision_render(void *filter_ptr, gs_effect_t *)
{
    struct vision_data *filter = filter_ptr;

    obs_source_t *target = obs_filter_get_target(filter->context);
    const obs_source_t *parent = obs_filter_get_parent(filter->context);
    if (!target || !parent || !filter->effect) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    const uint32_t width = obs_source_get_base_width(target);
    const uint32_t height = obs_source_get_base_height(target);
    if (!width || !height) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    if (filter->render_tick != filter->tick) {
        const bool resumed = filter->tick - filter->render_tick > 1;
        filter->render_tick = filter->tick;
        process_frame(filter, width, height, resumed);
    }

    /* Until there is a mask, draw nothing. Passing the frame through would show the background we're meant to hide. */
    if (!filter->have_mask || !filter->mask_texture)
        return;

    const struct frame_slot *slot = filter->sync ? filter->ready : filter->latest;
    gs_texture_t *texture = slot ? gs_texrender_get_texture(slot->texrender) : NULL;
    if (!texture)
        return;

    const float half_width = fmaxf(filter->softness, 0.002f) / 2.0f;
    struct vec2 mask_texel;
    vec2_set(&mask_texel, 1.0f / (float) gs_texture_get_width(filter->mask_texture),
             1.0f / (float) gs_texture_get_height(filter->mask_texture));

    const bool linear_srgb = gs_get_linear_srgb();
    const bool previous_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(linear_srgb);

    if (linear_srgb) {
        gs_effect_set_texture_srgb(filter->src_param, texture);
    } else {
        gs_effect_set_texture(filter->src_param, texture);
    }
    gs_effect_set_texture(filter->mask_param, filter->mask_texture);
    gs_effect_set_float(filter->mask_low_param, filter->threshold - half_width);
    gs_effect_set_float(filter->mask_high_param, filter->threshold + half_width);
    gs_effect_set_vec2(filter->mask_texel_param, &mask_texel);
    gs_effect_set_float(filter->range_scale_param, 400.0f * filter->refinement * filter->refinement);

    while (gs_effect_loop(filter->effect, "Draw")) {
        gs_draw_sprite(texture, 0, width, height);
    }

    gs_enable_framebuffer_srgb(previous_srgb);
}

static obs_properties_t *vision_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_property_t *list = obs_properties_add_list(props, "quality", obs_module_text("Quality"), OBS_COMBO_TYPE_LIST,
                                                   OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(list, obs_module_text("Quality.Accurate"),
                              VNGeneratePersonSegmentationRequestQualityLevelAccurate);
    obs_property_list_add_int(list, obs_module_text("Quality.Balanced"),
                              VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_property_list_add_int(list, obs_module_text("Quality.Fast"),
                              VNGeneratePersonSegmentationRequestQualityLevelFast);
    obs_property_t *sync = obs_properties_add_bool(props, "sync", obs_module_text("Sync"));
    obs_property_set_long_description(sync, obs_module_text("Sync.Description"));
    obs_properties_add_float_slider(props, "edge_threshold", obs_module_text("Threshold"), 0.05, 0.95, 0.01);
    obs_properties_add_float_slider(props, "edge_softness", obs_module_text("Softness"), 0, 1, 0.01);
    obs_properties_add_float_slider(props, "edge_refinement", obs_module_text("Refinement"), 0, 1, 0.01);
    return props;
}

static void vision_defaults(obs_data_t *settings)
{
    obs_data_set_default_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelAccurate);
    obs_data_set_default_bool(settings, "sync", true);
    obs_data_set_default_double(settings, "edge_threshold", 0.5);
    obs_data_set_default_double(settings, "edge_softness", 1.0);
    obs_data_set_default_double(settings, "edge_refinement", 0.5);
}

static void vision_update(void *filter_ptr, obs_data_t *settings)
{
    struct vision_data *filter = filter_ptr;

    filter->qualityLevel = obs_data_get_int(settings, "quality");
    filter->sync = obs_data_get_bool(settings, "sync");
    filter->threshold = (float) obs_data_get_double(settings, "edge_threshold");
    filter->softness = (float) obs_data_get_double(settings, "edge_softness");
    filter->refinement = (float) obs_data_get_double(settings, "edge_refinement");
}

static void *vision_create(obs_data_t *settings, struct obs_source *source)
{
    struct vision_data *filter = bzalloc(sizeof(struct vision_data));
    filter->context = source;
    pthread_mutex_init(&filter->result_mutex, NULL);

    /* The same request and sequence handler are used for every frame so that Vision treats the input as video. */
    filter->request = [[VNGeneratePersonSegmentationRequest alloc] init];
    filter->request.outputPixelFormat = kCVPixelFormatType_OneComponent8;
    filter->handler = [[VNSequenceRequestHandler alloc] init];

    /* Segmentation takes longer than a frame at higher quality levels, so it can't run on the graphics thread. */
    filter->mask_queue = dispatch_queue_create("Filter mask dispatch queue", DISPATCH_QUEUE_SERIAL);

    obs_enter_graphics();
    for (size_t i = 0; i < RING_SIZE; i++) {
        filter->slots[i].texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    }
    filter->analysis = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    char *file = obs_module_file("alpha_mask.effect");
    filter->effect = gs_effect_create_from_file(file, NULL);
    bfree(file);
    if (filter->effect) {
        filter->src_param = gs_effect_get_param_by_name(filter->effect, "image");
        filter->mask_param = gs_effect_get_param_by_name(filter->effect, "mask");
        filter->mask_low_param = gs_effect_get_param_by_name(filter->effect, "mask_low");
        filter->mask_high_param = gs_effect_get_param_by_name(filter->effect, "mask_high");
        filter->mask_texel_param = gs_effect_get_param_by_name(filter->effect, "mask_texel");
        filter->range_scale_param = gs_effect_get_param_by_name(filter->effect, "range_scale");
    } else {
        obs_log(LOG_ERROR, "Failed to load alpha_mask.effect");
    }
    obs_leave_graphics();

    vision_update(filter, settings);
    return filter;
}

static void vision_destroy(void *filter_ptr)
{
    struct vision_data *filter = filter_ptr;

    /* Wait for any segmentation still running; its block uses the filter. */
    dispatch_sync(filter->mask_queue, ^ {
                  });
    dispatch_release(filter->mask_queue);

    [filter->request release];
    [filter->handler release];
    if (filter->result_mask)
        CVPixelBufferRelease(filter->result_mask);
    if (filter->pool)
        CVPixelBufferPoolRelease(filter->pool);
    pthread_mutex_destroy(&filter->result_mutex);

    obs_enter_graphics();
    for (size_t i = 0; i < RING_SIZE; i++) {
        gs_texrender_destroy(filter->slots[i].texrender);
    }
    gs_texrender_destroy(filter->analysis);
    if (filter->stagesurf) {
        gs_stagesurface_destroy(filter->stagesurf);
    }
    if (filter->mask_texture) {
        gs_texture_destroy(filter->mask_texture);
    }
    gs_effect_destroy(filter->effect);
    obs_leave_graphics();

    bfree(filter);
};

bool obs_module_load(void)
{
    struct obs_source_info info = {
        .id = "mac_vision_filter",
        .type = OBS_SOURCE_TYPE_FILTER,
        .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
        .get_name = vision_get_name,
        .create = vision_create,
        .destroy = vision_destroy,
        .video_tick = vision_tick,
        .video_render = vision_render,
        .get_defaults = vision_defaults,
        .get_properties = vision_properties,
        .update = vision_update,
    };
    obs_register_source(&info);
    obs_log(LOG_INFO, "Loaded successfully (version %s)", PLUGIN_VERSION);
    return true;
}
