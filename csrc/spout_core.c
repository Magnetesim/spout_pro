/* Headless port of spout.c. See spout_core.h. Comments of the form
 * "orig: ..." point at the matching spot in ../spout/spout.c. */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spout_core.h"

/* Pull the tables from the original repo under private names so this file
 * can be linked next to the original spout.c in the fidelity test. */
#define sintable sp_sintable
#define FONT6 sp_FONT6
#define FONT16 sp_FONT16
#include "sintable.h"
#include "font.h"

#define FR SP_FRAMERATE

uint64_t sp_seed_to_state(uint64_t seed)
{
	/* splitmix64 so nearby seeds give unrelated streams */
	uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	z ^= z >> 31;
	return z ? z : 1;
}

static inline int sp_rand(Spout *s)
{
	return sp_rng_next(&s->rng);
}

/* orig: pceFontPrintf with font type 2, fg 3, bg 0 (the only in-game style) */
static void sp_print(Spout *s, int px, int py, const char *fmt, ...)
{
	char buf[64];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);

	uint8_t *adr = s->vbuff + px + py * SP_W;
	for (const char *pC = buf; *pC; pC++) {
		unsigned char ch = (unsigned char)*pC;
		int i = (ch >= 0x20 && ch < 0x80) ? ch - 0x20 : 0;
		const unsigned char *sAdr = FONT6 + (i & 15) + (i >> 4) * 16 * 16;
		for (int y = 0; y < 6; y++) {
			unsigned char c = *sAdr;
			for (int x = 0; x < 4; x++) {
				*adr++ = (c & 0x80) ? 3 : 0;
				c <<= 1;
			}
			adr += SP_W - 4;
			sAdr += 16;
		}
		adr -= SP_W * 6 - 4;
	}
}

/* orig: initGrain / allocGrain / freeGrain, with indices instead of pointers */
static void init_grain(Spout *s)
{
	int i;
	for (i = 0; i < SP_MAX_GRAIN - 1; i++)
		s->grain[i].next = (int16_t)(i + 1);
	s->grain[i].next = -1;
	s->free_head = 0;
	s->use_head = -1;
}

static int16_t alloc_grain(Spout *s)
{
	int16_t cur = s->free_head;
	if (cur >= 0) {
		SpGrain *g = &s->grain[cur];
		s->free_head = g->next;
		g->next = s->use_head;
		g->prev = -1;
		if (g->next >= 0)
			s->grain[g->next].prev = cur;
		s->use_head = cur;
	}
	return cur;
}

static int16_t free_grain(Spout *s, int16_t cur)
{
	SpGrain *g = &s->grain[cur];
	int16_t next = g->next;
	if (next >= 0)
		s->grain[next].prev = g->prev;
	if (g->prev >= 0)
		s->grain[g->prev].next = next;
	else
		s->use_head = next;
	g->next = s->free_head;
	s->free_head = cur;
	return next;
}

/* orig: spout() */
static void emit(Spout *s, int t, int x, int y)
{
	if (s->vbuff2[t] == 0 && s->nGrain < SP_MAX_GRAIN) {
		for (unsigned i = 0; i < SP_GRAIN_FACTOR; i++) {
			int16_t gi = alloc_grain(s);
			SpGrain *pG = &s->grain[gi];
			pG->v.x = (int16_t)x;
			pG->v.y = (int16_t)y;
			pG->s.x = 0;
			pG->s.y = 0;
			pG->color = (uint8_t)((2 + (sp_rand(s) & 1)) + 4 + 64 * 3);
			pG->pos = (int16_t)t;
			s->vbuff2[t] = pG->color;
			s->v2g[t] = gi;
			s->nGrain++;
		}
	}
}

/* orig: sweep() */
static void sweep(Spout *s, uint8_t c1, uint8_t c2)
{
	int p = 4 + 128 * ((s->upperLine + 77) & 127);
	for (int i = 0; i < 120; i++, p++) {
		if (s->vbuff2[p] & 4) {
			int16_t g = s->v2g[p];
			if (g >= 0) { /* the original would crash on a missing grain */
				free_grain(s, g);
				s->nGrain--;
			}
			s->v2g[p] = -1;
		}
		s->vbuff2[p] = c1;
	}
	p += 8;
	if (p >= 128 * 128)
		p -= 128 * 128;
	memset(s->vbuff2 + p, c2, 120);
}

/* orig: the `if(mPos.y < 40 * 256) { ... }` block: scroll the world one line */
static void scroll_line(Spout *s)
{
	s->mPosY += 256;
	s->upperLine = (s->upperLine - 1) & 127;
	s->height++;

	if (s->height > 0) {
		s->score++;
		s->ev_scored++;
		if ((s->height & 127) == 0) {
			int b = (s->time + FR - 1) / FR * 10;
			s->score += b;
			s->ev_bonus += b;
			s->time += 60 * FR;
			if (s->time > 99 * FR)
				s->time = 99 * FR;
			sp_print(s, 4 * 5, 82, "%2d", (s->time + FR - 1) / FR);
		}
		/* The barrier sits 1 row above the ship when height hits 128k,
		 * so reaching 128k+1 means the ship got through it. */
		if (s->height > 128 && (s->height & 127) == 1)
			s->ev_barrier++;
		sp_print(s, 4 * 15, 82, "%4d", s->height % 10000);
	}

	if (s->upperLine == 111 && s->height > 0) {
		memset(s->vbuff2 + 128 * 108 + 4, 0x00, 120);
		memset(s->vbuff2 + 128 * 109 + 4, 0xd3, 120);
		memset(s->vbuff2 + 128 * 110 + 4, 0x00, 120);
	}

	int box_x = 20 - (s->height + 40) / 64;
	if (box_x < 4)
		box_x = 4;
	int box_y = 20 - (s->height + 40) / 64;
	if (box_y < 4)
		box_y = 4;

	{
		int x = 4 + (sp_rand(s) % box_x);
		int y = 4 + (sp_rand(s) % box_y);
		int p = ((s->upperLine - 20 - (sp_rand(s) & 7)) & 127) * 128;
		int x1 = 4 + (sp_rand(s) % (120 - x));
		for (int i = y; i > 0; i--) {
			if (p < 0)
				p += 128 * 128;
			p += x1;
			memset(s->vbuff2 + p, 0, x);
			p += x;
			p -= x1 + x + 128;
		}
	}

	sweep(s, 0x13, 0xd2);
}

static void draw_hud(Spout *s)
{
	memset(s->vbuff + 128, 0x03, 128);
	if (s->height > 0)
		sp_print(s, 0, 82, "time:%2d height:%4d score:%6d", (s->time + FR - 1) / FR,
		         s->height % 10000, s->score % 1000000);
	else
		sp_print(s, 0, 82, "time:%2d height:   0 score:%6d", (s->time + FR - 1) / FR,
		         s->score % 1000000);
}

/* orig: the game-start branch of `if(!(gamePhase & 1))` with gamePhase == 2 */
void sp_reset(Spout *s, uint64_t seed)
{
	static const int32_t gx0[5] = {-2, 2, -1, 1, 0};

	memset(s, 0, sizeof *s);
	s->rng = sp_seed_to_state(seed);
	memcpy(s->gx, gx0, sizeof gx0);

	s->score = 0;
	s->dispscore = 0;
	s->height = -58;
	s->time = 60 * FR;

	for (int i = 0; i < 128 * 128; i++)
		s->v2g[i] = -1;
	init_grain(s);
	s->nGrain = 0;

	memset(s->vbuff2, 0xd2, 128 * 128);
	memset(s->vbuff2, 0, 128 * 78);
	memset(s->vbuff2 + 128 * (128 - 32), 0, 128 * 32);
	memset(s->vbuff, 0, sizeof s->vbuff);

	for (int i = 0; i < 128; i++) {
		memset(s->vbuff2 + i * 128, 0x0b, 4);
		memset(s->vbuff2 + i * 128 + 124, 0x0b, 4);
	}

	s->mPosX = 40 * 256;
	s->mPosY = 0;
	s->mSpeedX = 0;
	s->mSpeedY = 0;
	s->mR = 256 + 2 * 224;

	s->dispPos = 0;
	s->upperLine = 0;
	s->gameover = 0;
	s->rollCount = 0;

	draw_hud(s);
}

void sp_reset_at(Spout *s, uint64_t seed, int height, int x, int y, int mR)
{
	sp_reset(s, seed);
	if (height < -57)
		height = -57;
	while (s->height < height)
		scroll_line(s);

	s->score = 0;
	s->dispscore = 0;
	s->time = 60 * FR;

	/* Park the ship in a small cleared pocket so it doesn't start inside rock. */
	const int R = 6;
	for (int dy = -R; dy <= R; dy++) {
		int row = y + dy;
		if (row < 0 || row >= SP_PLAY_H)
			continue;
		for (int dx = -R; dx <= R; dx++) {
			int col = x + dx;
			if (col < 4 || col > 123 || dx * dx + dy * dy > R * R)
				continue;
			s->vbuff2[((row + s->upperLine) & 127) * 128 + col] = 0;
		}
	}

	s->mPosX = (int64_t)x * 256 + 128;
	s->mPosY = (int64_t)y * 256 + 128;
	s->mSpeedX = 0;
	s->mSpeedY = 0;
	s->mR = mR & 1023;
	s->dispPos = s->upperLine;

	memset(s->vbuff, 0, sizeof s->vbuff);
	draw_hud(s);
}

/* Hit on a destructible pixel: count it for the shaping reward if it is
 * above the ship and within 20 columns of it. */
static inline void note_hit(Spout *s, int pos)
{
	int row = ((pos >> 7) - s->upperLine) & 127;
	if (row >= SP_PLAY_H)
		row -= 128; /* hidden rows just above the top of the screen */
	int dx = (pos & 127) - (int)(s->mPosX / 256);
	if (row < s->mPosY / 256 && dx >= -20 && dx <= 20)
		s->ev_hits_above++;
}

/* Shared bounce handling for a grain that ran into a non-grain pixel.
 * orig: the `else` branches inside the four movement loops. */
static inline int hit_solid(Spout *s, SpGrain *pG, uint8_t *c)
{
	if (*c & 0xc0) {
		note_hit(s, (int)(c - s->vbuff2));
		*c -= 0x40;
		if (!(*c & 0xc0))
			*c = 0;
	}
	if (pG->color & 0xc0) {
		pG->color -= 0x40;
		return 0;
	}
	pG->color = 0;
	return 1;
}

static inline void swap_v(SpGrain *a, SpGrain *b)
{
	SpSVec t = a->v;
	a->v = b->v;
	b->v = t;
}

static void update_grains(Spout *s)
{
	int16_t gi = s->use_head;
	while (gi >= 0) {
		SpGrain *pG = &s->grain[gi];
		int f = 0;
		uint8_t *c;

		pG->v.y += 8;
		pG->s.x += pG->v.x;
		pG->s.y += pG->v.y;

		s->vbuff2[pG->pos] = 0;
		s->v2g[pG->pos] = -1;

		if (pG->s.y >= 256) {
			do {
				pG->s.y -= 256;
				pG->pos = (pG->pos + 128) & 16383;
				c = s->vbuff2 + pG->pos;
				if (*c) {
					if (*c & 0x04) {
						int16_t g2 = s->v2g[pG->pos];
						int r = 31 - (sp_rand(s) & 63);
						if (g2 >= 0) {
							SpGrain *pG2 = &s->grain[g2];
							swap_v(pG, pG2);
							pG->v.x += r;
							pG2->v.x -= r;
						}
					} else {
						pG->v.y = -pG->v.y / 2;
						pG->v.x += 15 - (sp_rand(s) & 31);
						f = hit_solid(s, pG, c);
					}
					pG->pos = (pG->pos - 128) & 16383;
					break;
				}
			} while (pG->s.y >= 256);
		} else {
			while (pG->s.y <= -256) {
				pG->s.y += 256;
				pG->pos = (pG->pos - 128) & 16383;
				c = s->vbuff2 + pG->pos;
				if (*c) {
					if (*c & 4) {
						if (s->v2g[pG->pos] >= 0)
							swap_v(pG, &s->grain[s->v2g[pG->pos]]);
					} else {
						pG->v.y = -pG->v.y / 2;
						f = hit_solid(s, pG, c);
					}
					pG->pos = (pG->pos + 128) & 16383;
					break;
				}
			}
		}

		if (pG->s.x >= 256) {
			do {
				pG->s.x -= 256;
				pG->pos = (pG->pos + 1) & 16383;
				c = s->vbuff2 + pG->pos;
				if (*c) {
					if (*c & 4) {
						if (s->v2g[pG->pos] >= 0)
							swap_v(pG, &s->grain[s->v2g[pG->pos]]);
					} else {
						pG->v.x = -pG->v.x / 2;
						f = hit_solid(s, pG, c);
					}
					pG->pos = (pG->pos - 1) & 16383;
					break;
				}
			} while (pG->s.x >= 256);
		} else {
			while (pG->s.x <= -256) {
				pG->s.x += 256;
				pG->pos = (pG->pos - 1) & 16383;
				c = s->vbuff2 + pG->pos;
				if (*c) {
					if (*c & 4) {
						if (s->v2g[pG->pos] >= 0)
							swap_v(pG, &s->grain[s->v2g[pG->pos]]);
					} else {
						pG->v.x = -pG->v.x / 2;
						f = hit_solid(s, pG, c);
					}
					pG->pos = (pG->pos + 1) & 16383;
					break;
				}
			}
		}

		if (f) {
			s->vbuff2[pG->pos] = pG->color;
			s->nGrain--;
			s->v2g[pG->pos] = -1;
			gi = free_grain(s, gi);
		} else {
			s->vbuff2[pG->pos] = pG->color;
			s->v2g[pG->pos] = gi;
			gi = pG->next;
		}
	}
}

static inline void put_px(Spout *s, int64_t x, int64_t y)
{
	int64_t idx = x / 256 + (y / 256 + 2) * 128;
	if (idx >= 0 && idx < (int64_t)sizeof s->vbuff)
		s->vbuff[idx] = 3;
}

/* orig: pceAppProc() with gamePhase == 3 */
void sp_frame(Spout *s, int pad)
{
	s->ev_scored = 0;
	s->ev_bonus = 0;
	s->ev_barrier = 0;
	s->ev_hits_above = 0;

	if (s->gameover == 0) {
		if (pad & SP_PAD_RI)
			s->mR = (s->mR - 16) & 1023;
		else if (pad & SP_PAD_LF)
			s->mR = (s->mR + 16) & 1023;
		if (pad & SP_PAD_A) {
			s->mSpeedX -= sintable[(256 + s->mR) & 1023] / SP_SPEED_FACTOR;
			s->mSpeedY += sintable[s->mR] / SP_SPEED_FACTOR;
		}
		s->mSpeedY += 8;

		if (s->mSpeedX < -SP_MAX_SPEED * 4)
			s->mSpeedX = -SP_MAX_SPEED * 4;
		else if (s->mSpeedX > SP_MAX_SPEED * 4)
			s->mSpeedX = SP_MAX_SPEED * 4;
		if (s->mSpeedY < -SP_MAX_SPEED * 4)
			s->mSpeedY = -SP_MAX_SPEED * 4;
		else if (s->mSpeedY > SP_MAX_SPEED * 4)
			s->mSpeedY = SP_MAX_SPEED * 4;

		s->mPosX += s->mSpeedX / 16;
		s->mPosY += s->mSpeedY / 16;

		if (s->mPosX >= 125 * 256) {
			s->mPosX = 124 * 256;
			s->gameover = 1;
		} else if (s->mPosX <= 2 * 256) {
			s->mPosX = 3 * 256;
			s->gameover = 1;
		}
		if (s->mPosY >= 78 * 256) {
			s->mPosY = 77 * 256;
			s->gameover = 1;
		}

		if (s->mPosY < 40 * 256)
			scroll_line(s);
	}

	s->rollCount++;

	{
		int r = sp_rand(s) & 3;
		int t = s->gx[r];
		s->gx[r] = s->gx[r + 1];
		s->gx[r + 1] = t;
		if (s->gameover == 0 && (pad & SP_PAD_A)) {
			for (int i = 0; i < 5; i++) {
				int g = s->gx[i];
				int pos = (int)(s->mPosX / 256) + g +
				          (int)(((s->mPosY / 256 - 1 + abs(g) + s->dispPos) & 127) * 128);
				int x = (int)(s->mSpeedX / 16 + sintable[(256 + s->mR) & 1023] / 8);
				int y = (int)(s->mSpeedY / 16 - sintable[s->mR] / 8);
				emit(s, pos, x, y);
			}
		}
	}

	update_grains(s);

	s->dispPos = s->upperLine;

	for (int r = 0; r < SP_PLAY_H; r++) {
		const uint8_t *src = s->vbuff2 + ((s->dispPos + r) & 127) * 128;
		uint8_t *dst = s->vbuff + (r + SP_PLAY_Y0) * SP_W;
		for (int x = 0; x < 128; x++)
			dst[x] = src[x] & 0x03;
	}

	{
		uint8_t c = s->vbuff2[s->mPosX / 256 + ((s->mPosY / 256 + s->dispPos) & 127) * 128];
		if (c != 0 && (c & 4) == 0)
			s->gameover = c;
	}

	if (s->gameover == 0) {
		int64_t x = s->mPosX + (int64_t)sintable[(256 + s->mR) & 1023] * s->gPhase / 64;
		int64_t y = s->mPosY - (int64_t)sintable[s->mR] * s->gPhase / 64;
		for (int i = 0; i < 3; i++) {
			if (y >= 78 * 256)
				break;
			put_px(s, x, y);
			x += sintable[(256 + s->mR) & 1023] / 16;
			y -= sintable[s->mR] / 16;

			if (y >= 78 * 256)
				break;
			put_px(s, x, y);
			x += sintable[(256 + s->mR) & 1023] / 16;
			y -= sintable[s->mR] / 16;

			if (y >= 78 * 256)
				break;
			put_px(s, x, y);
			x += sintable[(256 + s->mR) & 1023] * 2 / 16;
			y -= sintable[s->mR] * 2 / 16;
		}
		s->gPhase = (s->gPhase + 1) & 15;
	}

	{
		uint8_t *pC = s->vbuff + s->mPosX / 256 + (s->mPosY / 256 + 2) * 128;
		pC[-129] = 3;
		pC[-128] = 3;
		pC[-127] = 3;
		pC[-1] = 3;
		pC[0] = 0;
		pC[1] = 3;
		pC[127] = 3;
		pC[128] = 3;
		pC[129] = 3;
	}

	if (s->time && s->gameover == 0) {
		s->time--;
		if ((s->time % FR) == 0)
			sp_print(s, 4 * 5, 82, "%2d", (s->time + FR - 1) / FR);
		if (s->time == 0)
			s->gameover = 1;
	}

	if (s->dispscore < s->score) {
		s->dispscore++;
		if (s->dispscore < s->score)
			s->dispscore++;
		sp_print(s, 4 * 26, 82, "%6d", s->dispscore % 1000000);
	}

	if (s->gameover != 0)
		sp_print(s, 64 - 11 * 4 / 2, 33, " game over ");
}
