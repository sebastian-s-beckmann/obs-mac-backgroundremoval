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
#include <stdatomic.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

struct vision_data {
    obs_source_t *context;
    gs_texrender_t *texrender;
    gs_stagesurf_t *stagesurf;
    VNGeneratePersonSegmentationRequest *request;
    gs_effect_t *effect;
    gs_eparam_t *src_param;
    gs_eparam_t *mask_param;
    gs_eparam_t *threshold_param;
    float threshold;

    VNGeneratePersonSegmentationRequestQualityLevel qualityLevel;
    gs_texture_t *mask_texture;

    dispatch_queue_t mask_queue;
    pthread_mutex_t pixelBufferMutex;
    CVPixelBufferRef pixelBufferOut;

    atomic_bool in_flight;
    atomic_bool destroyed;
};

static const char *vision_get_name(void *)
{
    return obs_module_text("Name");
}

static void vision_render(void *filter_ptr, gs_effect_t *)
{
    struct vision_data *filter = filter_ptr;

    /* P0-2: Don't touch any state once teardown has begun */
    if (atomic_load(&filter->destroyed)) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    /* STEP ZERO: Prepare texrender */
    gs_texrender_reset(filter->texrender);

    obs_source_t *target = obs_filter_get_target(filter->context);
    const obs_source_t *parent = obs_filter_get_parent(filter->context);

    const uint32_t target_flags = obs_source_get_output_flags(target);

    const bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
    const bool async_source = (target_flags & OBS_SOURCE_ASYNC) != 0;

    const uint32_t width = obs_source_get_base_width(target);
    const uint32_t height = obs_source_get_base_height(target);

    /* P0-5: Guard zero-size source before any surface/buffer creation */
    if (width == 0 || height == 0) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    /* STEP ONE: Retrieve texture */
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    if (gs_texrender_begin(filter->texrender, width, height)) {
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0, 0);
        gs_ortho(0, width, 0, height, -100, 100);
        if (target == parent && !custom_draw && !async_source) {
            obs_source_default_render(target);
        } else {
            obs_source_video_render(target);
        }
        gs_texrender_end(filter->texrender);
    }
    gs_blend_state_pop();

    /* STEP TWO: Get source texture */
    gs_texture_t *source_texture = gs_texrender_get_texture(filter->texrender);
    if (!source_texture) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    enum gs_color_format format = gs_texture_get_color_format(source_texture);

    /* STEP THREE: Creation of new mask */
    /* P0-4: single-flight — only stage + dispatch a new request when none is in flight.
	 * Frames arriving while busy are dropped (bounded mask staleness). */
    if (!atomic_load(&filter->in_flight)) {
        /* STEP THREE point one: Stage source texture into an owned pixel buffer */
        if (filter->stagesurf && (width != gs_stagesurface_get_width(filter->stagesurf) ||
                                  height != gs_stagesurface_get_height(filter->stagesurf))) {
            gs_stagesurface_destroy(filter->stagesurf);
            filter->stagesurf = NULL;
        }
        if (!filter->stagesurf) {
            filter->stagesurf = gs_stagesurface_create(width, height, format);
        }

        uint8_t *data;
        uint32_t linesize;
        if (filter->stagesurf) {
            gs_stage_texture(filter->stagesurf, source_texture);
            if (gs_stagesurface_map(filter->stagesurf, &data, &linesize)) {
                /* P0-3: copy the staged bytes into an OWNED CVPixelBuffer while the
				 * surface is still mapped, so the async request never reads unmapped memory. */
                CVPixelBufferRef pixelBufferIn = NULL;
                CVReturn cr = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, NULL,
                                                  &pixelBufferIn);
                /* P1-1: also verify the input buffer locks before reading/writing its base
				 * address; a failed lock leaves the base address invalid. On failure, free the
				 * owned buffer, unmap, and skip staging this frame (no new request dispatched). */
                if (cr == kCVReturnSuccess && pixelBufferIn &&
                    CVPixelBufferLockBaseAddress(pixelBufferIn, 0) == kCVReturnSuccess) {
                    uint8_t *dst = CVPixelBufferGetBaseAddress(pixelBufferIn);
                    const size_t dst_stride = CVPixelBufferGetBytesPerRow(pixelBufferIn);
                    const size_t copy_bytes = (size_t) width * 4; /* 32BGRA */
                    for (uint32_t y = 0; y < height; y++) {
                        memcpy(dst + (size_t) y * dst_stride, data + (size_t) y * linesize, copy_bytes);
                    }
                    CVPixelBufferUnlockBaseAddress(pixelBufferIn, 0);
                    gs_stagesurface_unmap(filter->stagesurf);

                    /* P0-6: snapshot quality BY VALUE on the graphics thread */
                    const VNGeneratePersonSegmentationRequestQualityLevel quality = filter->qualityLevel;

                    /* STEP THREE point two: Dispatch creation of new mask */
                    if (!atomic_exchange(&filter->in_flight, true)) {
                        dispatch_async(filter->mask_queue, ^{
                            if (atomic_load(&filter->destroyed)) {
                                CVPixelBufferRelease(pixelBufferIn);
                                return;
                            }
                            filter->request.qualityLevel = quality;

                            VNImageRequestHandler *handler =
                                [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBufferIn options:@ {}];
                            NSArray *requests = @[filter->request];

                            /* P1-2: capture and check the Vision error instead of passing error:nil.
							 * `err` is an autoreleased NSError owned by the framework — no release here. */
                            NSError *err = nil;
                            BOOL ok = [handler performRequests:requests error:&err];

                            if (ok && filter->request.results.firstObject.pixelBuffer) {
                                CVPixelBufferRef mask = filter->request.results.firstObject.pixelBuffer;
                                pthread_mutex_lock(&filter->pixelBufferMutex);
                                if (filter->pixelBufferOut)
                                    CVPixelBufferRelease(filter->pixelBufferOut);
                                filter->pixelBufferOut = mask;
                                CVPixelBufferRetain(filter->pixelBufferOut);
                                pthread_mutex_unlock(&filter->pixelBufferMutex);
                            } else if (!ok) {
                                /* P1-2: request failed — keep the last good mask (leave
								 * pixelBufferOut untouched so the background does not flash back in). */
                                obs_log(LOG_WARNING, "Vision request failed: %s",
                                        err ? err.localizedDescription.UTF8String : "unknown error");
                            } else {
                                /* P1-2: request succeeded but returned no mask — keep last good mask. */
                                obs_log(LOG_WARNING, "Vision request returned no segmentation mask");
                            }

                            CVPixelBufferRelease(pixelBufferIn);
                            [handler release];
                            atomic_store(&filter->in_flight, false);
                        });
                    } else {
                        /* Lost the single-flight race; drop this frame */
                        CVPixelBufferRelease(pixelBufferIn);
                    }
                } else {
                    /* P1-1: input buffer create or lock failed — release any owned buffer and
					 * unmap; skip this frame (the last good mask, if any, keeps applying). */
                    obs_log(LOG_WARNING, "Failed to create/lock input pixel buffer; skipping frame");
                    if (pixelBufferIn)
                        CVPixelBufferRelease(pixelBufferIn);
                    gs_stagesurface_unmap(filter->stagesurf);
                }
            }
        } else {
            /* P1-1: staging surface creation failed; can't stage this frame. The existing
			 * structural guard already prevents use; add a log so the failure is visible. */
            obs_log(LOG_WARNING, "Failed to create staging surface (%ux%u)", width, height);
        }
    }

    /* STEP FOUR: Retrieve new mask texture */
    /* P0-6: read pixelBufferOut + build the mask texture ENTIRELY under the mutex */
    pthread_mutex_lock(&filter->pixelBufferMutex);

    /* Don't render if the output pixel buffer doesn't exist, like before the first frame is processed */
    if (!filter->pixelBufferOut) {
        pthread_mutex_unlock(&filter->pixelBufferMutex);
        obs_source_skip_video_filter(filter->context);
        return;
    }

    /* STEP FOUR point one: Prepare mask texture */
    /* P1-1: a failed read-only lock leaves the base address invalid; unlock the mutex and
	 * skip this frame rather than read garbage. The last good mask remains for next frame. */
    if (CVPixelBufferLockBaseAddress(filter->pixelBufferOut, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
        pthread_mutex_unlock(&filter->pixelBufferMutex);
        obs_source_skip_video_filter(filter->context);
        return;
    }

    const uint32_t out_width = (uint32_t) CVPixelBufferGetWidth(filter->pixelBufferOut);
    const uint32_t out_height = (uint32_t) CVPixelBufferGetHeight(filter->pixelBufferOut);
    if (filter->mask_texture && (out_width != gs_texture_get_width(filter->mask_texture) ||
                                 out_height != gs_texture_get_height(filter->mask_texture))) {
        gs_texture_destroy(filter->mask_texture);
        filter->mask_texture = NULL;
    }

    if (!filter->mask_texture) {
        filter->mask_texture =
            gs_texture_create((uint32_t) out_width, (uint32_t) out_height, GS_A8, 1, NULL, GS_DYNAMIC);
    }

    /* STEP FOUR point two: Get mask texture from pixel buffer */
    const uint8_t *base_address = CVPixelBufferGetBaseAddress(filter->pixelBufferOut);
    const uint32_t bytes_per_row = (uint32_t) CVPixelBufferGetBytesPerRow(filter->pixelBufferOut);
    gs_texture_set_image(filter->mask_texture, base_address, bytes_per_row, false);

    CVPixelBufferUnlockBaseAddress(filter->pixelBufferOut, kCVPixelBufferLock_ReadOnly);
    pthread_mutex_unlock(&filter->pixelBufferMutex);

    /* STEP FIVE: Render result */
    if (obs_source_process_filter_begin(filter->context, format, OBS_ALLOW_DIRECT_RENDERING)) {
        gs_effect_set_texture_srgb(filter->src_param, source_texture);
        gs_effect_set_texture_srgb(filter->mask_param, filter->mask_texture);
        gs_effect_set_float(filter->threshold_param, filter->threshold);

        gs_blend_state_push();
        obs_source_process_filter_tech_end(filter->context, filter->effect, 0, 0, "Draw");
        gs_blend_state_pop();
    }
}

static obs_properties_t *vision_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_float_slider(props, "threshold", obs_module_text("Threshold"), 0, 1, 0.05);
    obs_property_t *list = obs_properties_add_list(props, "quality", obs_module_text("Quality"), OBS_COMBO_TYPE_LIST,
                                                   OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(list, obs_module_text("Quality.Accurate"),
                              VNGeneratePersonSegmentationRequestQualityLevelAccurate);
    obs_property_list_add_int(list, obs_module_text("Quality.Balanced"),
                              VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_property_list_add_int(list, obs_module_text("Quality.Fast"),
                              VNGeneratePersonSegmentationRequestQualityLevelFast);
    return props;
}

static void vision_defaults(obs_data_t *settings)
{
    obs_data_set_default_double(settings, "threshold", 0.9);
    obs_data_set_default_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelBalanced);
}

static void vision_update(void *filter_ptr, obs_data_t *settings)
{
    struct vision_data *filter = filter_ptr;

    filter->threshold = obs_data_get_double(settings, "threshold");
    filter->qualityLevel = obs_data_get_int(settings, "quality");
}

static void *vision_create(obs_data_t *settings, struct obs_source *source)
{
    struct vision_data *filter = bzalloc(sizeof(struct vision_data));
    filter->context = source;
    filter->request = [[VNGeneratePersonSegmentationRequest alloc] init];
    pthread_mutex_init(&filter->pixelBufferMutex, NULL);
    atomic_init(&filter->in_flight, false);
    atomic_init(&filter->destroyed, false);

    /* Performing the segmentation in realtime takes too long and will lag OBS, especially at higher
	 * quality modes. As such, defer it to a different thread. This will make the mask lag behind a few
	 * frames, but is better than lagging the graphics thread. */
    filter->mask_queue = dispatch_queue_create("Filter mask dispatch queue", NULL);

    obs_enter_graphics();
    filter->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    char *file = obs_module_file("alpha_mask.effect");
    filter->effect = gs_effect_create_from_file(file, NULL);
    bfree(file);

    /* P1-1: If the shader fails to compile/load, gs_effect_create_from_file returns NULL.
	 * The eparam lookups below would then dereference a NULL effect, and vision_render would
	 * later NULL-deref the params. Fail vision_create cleanly so OBS refuses to attach the
	 * filter rather than crash later. Tear down only what has been allocated up to here:
	 * graphics objects inside graphics (effect is NULL; texrender may exist), then the ObjC
	 * request, the mutex, and the dispatch queue outside graphics, then bfree last. */
    if (!filter->effect) {
        if (filter->texrender) {
            gs_texrender_destroy(filter->texrender);
        }
        gs_effect_destroy(filter->effect);
        obs_leave_graphics();

        obs_log(LOG_ERROR, "Failed to load alpha_mask.effect; filter will not attach");

        [filter->request release];
        pthread_mutex_destroy(&filter->pixelBufferMutex);
        dispatch_release(filter->mask_queue);
        bfree(filter);
        return NULL;
    }

    filter->src_param = gs_effect_get_param_by_name(filter->effect, "image");
    filter->mask_param = gs_effect_get_param_by_name(filter->effect, "mask");
    filter->threshold_param = gs_effect_get_param_by_name(filter->effect, "threshold");
    obs_leave_graphics();
    vision_update(filter, settings);
    return filter;
}

static void vision_destroy(void *filter_ptr)
{
    struct vision_data *filter = filter_ptr;

    /* P0-2: Signal teardown, then drain the serial queue so any in-flight Vision
	 * block finishes (and releases its owned input buffer) before we free anything.
	 * Run the drain OUTSIDE obs_enter_graphics() — never hold the graphics lock
	 * across the (slow) Vision completion. */
    atomic_store(&filter->destroyed, true);
    dispatch_sync(filter->mask_queue, ^ {
                  });

    /* GPU objects — graphics thread only */
    obs_enter_graphics();
    gs_texrender_destroy(filter->texrender);
    if (filter->stagesurf) {
        gs_stagesurface_destroy(filter->stagesurf);
    }
    if (filter->mask_texture) {
        gs_texture_destroy(filter->mask_texture);
    }
    gs_effect_destroy(filter->effect);
    obs_leave_graphics();

    /* P0-2: release the resources the old teardown leaked */
    [filter->request release];
    if (filter->pixelBufferOut) {
        CVPixelBufferRelease(filter->pixelBufferOut);
    }
    pthread_mutex_destroy(&filter->pixelBufferMutex);
    dispatch_release(filter->mask_queue);

    bfree(filter);
}

bool obs_module_load(void)
{
    struct obs_source_info info = {
        .id = "mac_vision_filter",
        .type = OBS_SOURCE_TYPE_FILTER,
        .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
        .get_name = vision_get_name,
        .create = vision_create,
        .destroy = vision_destroy,
        .video_render = vision_render,
        .get_defaults = vision_defaults,
        .get_properties = vision_properties,
        .update = vision_update,
    };
    obs_register_source(&info);
    obs_log(LOG_INFO, "Loaded successfully (version %s)", PLUGIN_VERSION);
    return true;
}
