//
// Created by lovemefan on 2024/7/19.
//

#ifndef SENSEVOICE_CPP_SENSE_VOICE_H
#define SENSEVOICE_CPP_SENSE_VOICE_H

#include "common.h"
#include "sense-voice-encoder.h"




int sense_voice_lang_id(const char * lang);
const char * sense_voice_lang_str(int id);
struct sense_voice_context_params sense_voice_context_default_params();
struct sense_voice_context * sense_voice_small_init_from_file_with_params(const char * path_model, struct sense_voice_context_params params);
struct sense_voice_context * sense_voice_small_init_from_file_with_params_no_state(const char * path_model, struct sense_voice_context_params params);
struct sense_voice_context *sense_voice_init_with_params_no_state(const char *path_model, sense_voice_context_params params);
int sense_voice_full_parallel(struct sense_voice_context * ctx,
                              sense_voice_full_params &params,
                              std::vector<double> &samples,
                              int n_samples,
                              int n_processors);
void sense_voice_print_output(struct sense_voice_context * ctx, bool need_prefix, bool use_itn, bool refresh_self=false);
void sense_voice_free_state(struct sense_voice_state * state);
int sense_voice_batch_full(struct sense_voice_context * ctx, const sense_voice_full_params &params);
int sense_voice_batch_pcmf(struct sense_voice_context *ctx, const sense_voice_full_params &params, std::vector<std::vector<float>> &pcmf32,
                           size_t max_batch_len=90000, size_t max_batch_cnt=1,
                           bool use_prefix=true, bool use_itn=true);
void sense_voice_batch_print_output(struct sense_voice_context * ctx, bool need_prefix, bool use_itn, bool refresh_self=false);

/**
 * Run inference on pre-computed features (decoupled from PCM feature extraction).
 * Adapted from sherpa-onnx's composable Run() API.
 *
 * @param ctx SenseVoice context
 * @param params Decoding parameters
 * @param features Pre-computed features of shape [n_frames, feat_dim] where feat_dim = lfr_m * n_mels (default 560)
 * @param n_frames Number of frames
 * @param feat_dim Feature dimension (default 560 after LFR)
 * @param language Language ID (0=auto, 3=zh, 4=en, 7=yue, 11=ja, 12=ko)
 * @param text_norm ITN control (14=with ITN, 15=without ITN)
 * @returns 0 on success, -1 on failure
 */
int sense_voice_run_features(struct sense_voice_context *ctx,
                             const sense_voice_full_params &params,
                             const std::vector<float> &features,
                             int n_frames,
                             int feat_dim,
                             int language = 0,
                             int text_norm = 14);

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
struct sense_voice_context * sense_voice_small_init_from_android_asset(
    AAssetManager *mgr, const char *path_model, struct sense_voice_context_params params);
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
struct sense_voice_context * sense_voice_small_init_from_harmonyos_asset(
    NativeResourceManager *mgr, const char *path_model, struct sense_voice_context_params params);
#endif

#endif//SENSEVOICE_CPP_SENSE_VOICE_H
