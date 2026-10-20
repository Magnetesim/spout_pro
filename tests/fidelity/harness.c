/* Fidelity test: run the original spout.c and the headless port side by side
 * with the same random numbers and the same buttons, and check that the
 * screen (vbuff), the world buffer (vbuff2) and the ship state match on
 * every frame.
 *
 * The original is compiled from a copy with two small edits made by the
 * Makefile: rand()/srand() go through a shared generator, and the static
 * locals gx/gPhase become globals so the port can be synced to them. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spout_core.h"

uint64_t test_rng = 1;
int test_rand(void) { return sp_rng_next(&test_rng); }
void test_srand(unsigned s) { (void)s; }
int test_gx[5] = {-2, 2, -1, 1, 0};
int test_gPhase = 0;
unsigned char fake_keys[512];

#define rand test_rand
#define srand test_srand
#define main orig_main
#include "spout_patched.c"
#undef rand
#undef srand
#undef main
#undef gx
#undef gPhase

static uint64_t ctl_rng = 12345; /* harness choices, separate from the game */
static int ctl(int n) { return (int)(sp_rng_next(&ctl_rng) % (unsigned)n); }

static void set_keys(int pad)
{
	memset(fake_keys, 0, sizeof fake_keys);
	if (pad & SP_PAD_RI) fake_keys[SDLK_RIGHT] = SDL_PRESSED;
	if (pad & SP_PAD_LF) fake_keys[SDLK_LEFT] = SDL_PRESSED;
	if (pad & SP_PAD_A) fake_keys[SDLK_x] = SDL_PRESSED;
}

static int orig_cnt = 0;
static void orig_frame(int pad)
{
	set_keys(pad);
	keys = fake_keys;
	pceAppProc(orig_cnt++);
}

/* A jittery climber: mostly keeps the nose up and thrusts when falling,
 * with random spins and bursts so grains fly everywhere. */
static int pick_pad(const Spout *s, int *hold, int *cur)
{
	if (*hold > 0) {
		(*hold)--;
		return *cur;
	}
	int pad = 0;
	int mode = ctl(10);
	if (mode < 6) {
		int diff = ((768 + ctl(161) - 80 - s->mR) & 1023);
		if (diff > 16 && diff < 512) pad |= SP_PAD_LF;
		else if (diff >= 512 && diff < 1008) pad |= SP_PAD_RI;
		if (s->mSpeedY > -200 + ctl(300)) pad |= SP_PAD_A;
	} else {
		pad = (ctl(3) == 0 ? 0 : (ctl(2) ? SP_PAD_LF : SP_PAD_RI)) | (ctl(2) ? SP_PAD_A : 0);
	}
	*cur = pad;
	*hold = ctl(12);
	return pad;
}

static int compare(const Spout *s, int ep, int f)
{
	const char *what = NULL;
	if (memcmp(vbuff, s->vbuff, sizeof vbuff)) what = "vbuff";
	else if (memcmp(vbuff2, s->vbuff2, sizeof vbuff2)) what = "vbuff2";
	else if (mPos.x != s->mPosX || mPos.y != s->mPosY) what = "position";
	else if (mSpeed.x != s->mSpeedX || mSpeed.y != s->mSpeedY) what = "speed";
	else if (mR != s->mR) what = "angle";
	else if (height != s->height || score != s->score || time != s->time) what = "hud";
	else if (nGrain != s->nGrain) what = "nGrain";
	else if (test_rng != s->rng) what = "rng";
	if (what) {
		if (!strcmp(what, "vbuff")) for (int k = 0; k < (int)sizeof vbuff; k++) if (vbuff[k] != s->vbuff[k]) fprintf(stderr, "  px x=%d y=%d orig=%d port=%d\n", k % 128, k / 128, vbuff[k], s->vbuff[k]);
		fprintf(stderr, "  ship x=%ld y=%ld mR=%d go=%d time=%d\n", (long)(s->mPosX/256), (long)(s->mPosY/256), s->mR, s->gameover, s->time);
		fprintf(stderr, "MISMATCH ep %d frame %d: %s (orig h=%d score=%d, port h=%d score=%d)\n",
		        ep, f, what, height, score, s->height, s->score);
		return 1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	int episodes = argc > 1 ? atoi(argv[1]) : 200;
	static Spout s;
	long total_frames = 0;
	int max_height = 0, barriers = 0;

	setenv("HOME", "/nonexistent-spout-test", 1); /* never touch the real high-score file */
	initSDL();
	pceAppInit();

	orig_frame(0); /* title screen init */
	for (int ep = 0; ep < episodes; ep++) {
		for (int i = 0, n = 1 + ctl(20); i < n; i++)
			orig_frame(0); /* idle on the title screen */
		orig_frame(SP_PAD_A); /* press start: TRG_A edge */

		/* The next call runs the game-start code and then the first frame.
		 * Sync the port to the original's generator state first. */
		uint64_t rng_at_start = test_rng;
		int warp = (ep % 3 == 0) ? 0 : ctl(1500);
		int pad = 0, hold = 0, cur = 0, prev_pad = 0;

		sp_reset(&s, 0);
		s.rng = rng_at_start;
		memcpy(s.gx, test_gx, sizeof s.gx);
		s.gPhase = test_gPhase;

		int f = 0, after = 0;
		for (;; f++) {
			pad = s.gameover ? 0 : pick_pad(&s, &hold, &cur);
			if (f == 0) pad = 0; /* don't make a TRG_A edge on the start frame */
			/* Pressing thrust (a TRG_A edge) on the frame the ship dies makes the
			 * original skip straight to the title. Don't do that. */
			if ((pad & SP_PAD_A) && !(prev_pad & SP_PAD_A)) {
				static Spout peek;
				peek = s;
				sp_frame(&peek, pad);
				if (peek.gameover) pad &= ~SP_PAD_A;
			}
			prev_pad = pad;
			orig_frame(pad);
			sp_frame(&s, pad);
			if (f == 0 && warp) { /* jump both games to a deeper part of the cave */
				height = warp;
				s.height = warp;
			}
			if (compare(&s, ep, f)) return 1;
			if (s.height > max_height) max_height = s.height;
			barriers += s.ev_barrier;
			if (s.gameover && ++after > 30) break;
			if (f > 20000) break;
		}
		total_frames += f;

		/* Leave the game-over screen: release, then press to go back to title. */
		orig_frame(0);
		orig_frame(SP_PAD_A);
		orig_frame(0);
	}
	printf("OK: %d episodes, %ld frames identical (max height %d, barriers passed %d)\n",
	       episodes, total_frames, max_height, barriers);
	return 0;
}
