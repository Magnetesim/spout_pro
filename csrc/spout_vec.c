#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "spout_vec.h"
#include <omp.h>

static int action_pad(int32_t a)
{
	static const int rot[3] = {0, SP_PAD_LF, SP_PAD_RI};
	if (a < 0 || a >= SPV_NUM_ACTIONS)
		a = 0;
	return rot[a % 3] | ((a / 3) ? SP_PAD_A : 0);
}

static uint64_t next_u64(uint64_t *state)
{
	uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static int rand_range(uint64_t *state, int lo, int hi) /* inclusive */
{
	return lo + (int)(next_u64(state) % (uint64_t)(hi - lo + 1));
}

static float rand_unit(uint64_t *state)
{
	return (float)(next_u64(state) >> 40) / (float)(1 << 24);
}

static void write_obs(const Spout *s, uint8_t *obs, float *sc)
{
	memcpy(obs, s->vbuff + SP_PLAY_Y0 * SP_W, SP_PLAY_H * SP_W);
	/* Things a player can read off the screen: the HUD (time, height) plus
	 * the ship's position, heading and motion. */
	float ang = (float)s->mR * (2.0f * 3.14159265f / 1024.0f);
	sc[0] = (float)s->time / (99.0f * SP_FRAMERATE);
	sc[1] = (float)(s->height > 0 ? (s->height & 127) : 0) / 128.0f;
	sc[2] = (float)s->mPosX / (128.0f * 256.0f);
	sc[3] = (float)s->mPosY / (78.0f * 256.0f);
	sc[4] = sinf(ang);
	sc[5] = cosf(ang);
	sc[6] = (float)s->mSpeedX / (SP_MAX_SPEED * 4.0f);
	sc[7] = (float)s->mSpeedY / (SP_MAX_SPEED * 4.0f);
}

static void start_env(SpVec *v, int i, uint64_t seed, int mode, int height, int x, int y,
                      int mR)
{
	Spout *s = &v->envs[i];
	if (mode == SPV_START_NORMAL) {
		sp_reset(s, seed);
		height = 0;
	} else {
		sp_reset_at(s, seed, height, x, y, mR);
	}
	/* Show one real frame so the first observation isn't blank. */
	sp_frame(s, 0);
	v->ep_frames[i] = 1;
	v->ep_return[i] = 0.0f;
	v->ep_mode[i] = mode;
	v->ep_start_height[i] = height;
	v->ep_barriers[i] = 0;
}

static void random_start(SpVec *v, int i)
{
	uint64_t *r = &v->start_rng[i];
	uint64_t seed = next_u64(r);
	float u = rand_unit(r);
	const SpvConfig *c = &v->cfg;

	if (u < c->p_drill) {
		/* Barriers appear when height = 87 + 128k and then move down one
		 * row per height; after d more rows the barrier is on screen row d-2. */
		int k = rand_range(r, 0, c->drill_max_barrier > 0 ? c->drill_max_barrier - 1 : 0);
		int d = rand_range(r, 8, 36);
		start_env(v, i, seed, SPV_START_DRILL, 87 + 128 * k + d, rand_range(r, 16, 111),
		          rand_range(r, 50, 66), 768 + rand_range(r, -96, 96));
	} else if (u < c->p_drill + c->p_random) {
		start_env(v, i, seed, SPV_START_RANDOM, rand_range(r, 1, c->random_height_max),
		          rand_range(r, 16, 111), rand_range(r, 50, 66), 768 + rand_range(r, -96, 96));
	} else {
		start_env(v, i, seed, SPV_START_NORMAL, 0, 0, 0, 0);
	}
}

SpVec *spv_create(int32_t n, uint64_t seed, const SpvConfig *cfg)
{
	SpVec *v = calloc(1, sizeof *v);
	v->n = n;
	v->nthreads = 8;
	v->cfg = *cfg;
	v->envs = calloc((size_t)n, sizeof(Spout));
	v->start_rng = calloc((size_t)n, sizeof(uint64_t));
	v->ep_frames = calloc((size_t)n, sizeof(int32_t));
	v->ep_return = calloc((size_t)n, sizeof(float));
	v->ep_mode = calloc((size_t)n, sizeof(int32_t));
	v->ep_start_height = calloc((size_t)n, sizeof(int32_t));
	v->ep_barriers = calloc((size_t)n, sizeof(int32_t));
	for (int i = 0; i < n; i++)
		v->start_rng[i] = sp_seed_to_state(seed * 1000003ULL + (uint64_t)i);
	return v;
}

void spv_free(SpVec *v)
{
	if (!v)
		return;
	free(v->envs);
	free(v->start_rng);
	free(v->ep_frames);
	free(v->ep_return);
	free(v->ep_mode);
	free(v->ep_start_height);
	free(v->ep_barriers);
	free(v);
}

void spv_set_config(SpVec *v, const SpvConfig *cfg)
{
	v->cfg = *cfg;
}

void spv_reset_all(SpVec *v, uint8_t *obs, float *scalars)
{
#pragma omp parallel for schedule(dynamic, 4) num_threads(v->nthreads)
	for (int i = 0; i < v->n; i++) {
		random_start(v, i);
		write_obs(&v->envs[i], obs + (size_t)i * SP_PLAY_H * SP_W,
		          scalars + (size_t)i * SPV_NUM_SCALARS);
	}
}

void spv_start(SpVec *v, int32_t i, uint64_t seed, int32_t mode, int32_t height, int32_t x,
               int32_t y, int32_t mR, uint8_t *obs_i, float *scalars_i)
{
	start_env(v, i, seed, mode, height, x, y, mR);
	if (obs_i && scalars_i)
		write_obs(&v->envs[i], obs_i, scalars_i);
}

void spv_step(SpVec *v, const int32_t *actions, uint8_t *obs, float *scalars, float *rewards,
              uint8_t *dones, float *info, uint8_t *frames)
{
	const SpvConfig c = v->cfg;
#pragma omp parallel for schedule(dynamic, 4) num_threads(v->nthreads)
	for (int i = 0; i < v->n; i++) {
		Spout *s = &v->envs[i];
		int pad = action_pad(actions[i]);
		float r = 0.0f;
		int done = 0, capped = 0;

		for (int k = 0; k < c.frameskip; k++) {
			sp_frame(s, pad);
			v->ep_frames[i]++;
			r += c.r_height * (float)s->ev_scored + c.r_bonus * (float)s->ev_bonus +
			     c.r_barrier * (float)s->ev_barrier +
			     c.r_hit_above * (float)s->ev_hits_above;
			v->ep_barriers[i] += s->ev_barrier;
			if (frames)
				memcpy(frames + ((size_t)i * c.frameskip + k) * SP_SCREEN_H * SP_W,
				       s->vbuff, SP_SCREEN_H * SP_W);
			if (s->gameover) {
				r += c.r_death;
				done = 1;
				/* Pad the remaining frames with the game-over screen. */
				if (frames)
					for (int k2 = k + 1; k2 < c.frameskip; k2++)
						memcpy(frames + ((size_t)i * c.frameskip + k2) * SP_SCREEN_H * SP_W,
						       s->vbuff, SP_SCREEN_H * SP_W);
				break;
			}
			if (c.max_frames > 0 && v->ep_frames[i] >= c.max_frames) {
				done = 1;
				capped = 1;
				break;
			}
		}

		v->ep_return[i] += r;
		rewards[i] = r;
		dones[i] = (uint8_t)done;

		float *inf = info + (size_t)i * SPV_NUM_INFO;
		if (done) {
			inf[0] = v->ep_return[i];
			inf[1] = (float)s->height;
			inf[2] = (float)s->score;
			inf[3] = (float)v->ep_frames[i];
			inf[4] = (float)v->ep_mode[i];
			inf[5] = (float)v->ep_start_height[i];
			inf[6] = (float)v->ep_barriers[i];
			inf[7] = capped ? -1.0f : (s->time == 0 ? 1.0f : 0.0f);
			random_start(v, i);
		} else {
			memset(inf, 0, SPV_NUM_INFO * sizeof(float));
		}
		write_obs(s, obs + (size_t)i * SP_PLAY_H * SP_W, scalars + (size_t)i * SPV_NUM_SCALARS);
	}
}

void spv_get_state(SpVec *v, int32_t i, int64_t *out)
{
	const Spout *s = &v->envs[i];
	out[0] = s->mPosX;
	out[1] = s->mPosY;
	out[2] = s->mSpeedX;
	out[3] = s->mSpeedY;
	out[4] = s->mR;
	out[5] = s->height;
	out[6] = s->score;
	out[7] = s->time;
	out[8] = s->gameover;
	out[9] = s->nGrain;
}

int32_t spv_sizeof_spout(void)
{
	return (int32_t)sizeof(Spout);
}

void spv_set_threads(SpVec *v, int32_t nthreads)
{
	v->nthreads = nthreads > 0 ? nthreads : 1;
}
