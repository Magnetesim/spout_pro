/* Batched environment API used from Python via ctypes. */
#ifndef SPOUT_VEC_H
#define SPOUT_VEC_H

#include <stdint.h>
#include "spout_core.h"

#define SPV_NUM_ACTIONS 6 /* {none, left, right} x {coast, thrust} */
#define SPV_NUM_SCALARS 8
#define SPV_NUM_INFO 8

/* Start modes */
#define SPV_START_NORMAL 0
#define SPV_START_DRILL 1  /* barrier visible above the ship */
#define SPV_START_RANDOM 2 /* random height, ship in a pocket */

typedef struct {
	int32_t frameskip;  /* frames per decision */
	int32_t max_frames; /* episode cap (counts as a normal end, no death penalty) */

	float r_height;    /* per pixel of height while height > 0 */
	float r_bonus;     /* per point of the game's time bonus */
	float r_barrier;   /* per barrier passed */
	float r_death;     /* once, on game over (crash or time out) */
	float r_hit_above; /* per hp of terrain knocked off above the ship */

	float p_drill;  /* chance an episode is a barrier drill */
	float p_random; /* chance an episode starts at a random height */
	int32_t random_height_max;
	int32_t drill_max_barrier; /* drills use barriers 1..this */
} SpvConfig;

typedef struct {
	int32_t n;
	int32_t nthreads;
	SpvConfig cfg;
	Spout *envs;
	uint64_t *start_rng; /* per-env stream for picking starts and seeds */
	int32_t *ep_frames;
	float *ep_return;
	int32_t *ep_mode;
	int32_t *ep_start_height;
	int32_t *ep_barriers;
} SpVec;

/* Per-env info written by spv_step (only meaningful when done[i] == 1):
 * 0 return, 1 final height, 2 final score, 3 frames, 4 start mode,
 * 5 start height, 6 barriers passed, 7 timed out (1) / crashed (0) / capped (-1) */

SpVec *spv_create(int32_t n, uint64_t seed, const SpvConfig *cfg);
void spv_free(SpVec *v);
void spv_set_config(SpVec *v, const SpvConfig *cfg);

/* Reset every env using the start distribution and write the first observation. */
void spv_reset_all(SpVec *v, uint8_t *obs, float *scalars);

/* Start env i at an exact configuration (used for replays and evaluation). */
void spv_start(SpVec *v, int32_t i, uint64_t seed, int32_t mode, int32_t height,
               int32_t x, int32_t y, int32_t mR, uint8_t *obs_i, float *scalars_i);

/* Step all envs. Envs that finish are reset automatically, and obs then holds
 * the first frame of the new episode.
 * obs:     n * 78 * 128 bytes (play area, colors 0..3)
 * scalars: n * SPV_NUM_SCALARS
 * frames:  optional (may be NULL), n * frameskip * 88 * 128, every frame shown
 *          to the player during this step (full screen including HUD) */
void spv_step(SpVec *v, const int32_t *actions, uint8_t *obs, float *scalars, float *rewards,
              uint8_t *dones, float *info, uint8_t *frames);

/* Raw state of env i: x, y, vx, vy, mR, height, score, time, gameover, nGrain */
void spv_get_state(SpVec *v, int32_t i, int64_t *out);

int32_t spv_sizeof_spout(void);
void spv_set_threads(SpVec *v, int32_t nthreads);

#endif
