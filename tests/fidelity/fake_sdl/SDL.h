/* Just enough of SDL 1.2 to compile spout.c headless for the fidelity test. */
#ifndef FAKE_SDL_H
#define FAKE_SDL_H
#include <stdint.h>
#include <stdlib.h>

typedef uint8_t Uint8;
typedef uint32_t Uint32;
typedef struct { Uint8 r, g, b, unused; } SDL_Color;
typedef struct { Uint32 Rmask, Gmask, Bmask, Amask; } SDL_PixelFormat;
typedef struct { SDL_PixelFormat *format; int w, h, pitch; void *pixels; } SDL_Surface;
typedef struct { int16_t x, y; uint16_t w, h; } SDL_Rect;
typedef struct { Uint8 type; } SDL_Event;

#define SDL_INIT_VIDEO 0x20
#define SDL_SWSURFACE 0
#define SDL_HWSURFACE 1
#define SDL_DOUBLEBUF 2
#define SDL_HWPALETTE 4
#define SDL_FULLSCREEN 8
#define SDL_PRESSED 1
#define SDL_QUIT 12

enum {
	SDLK_RETURN = 13, SDLK_ESCAPE = 27, SDLK_SPACE = 32, SDLK_x = 120, SDLK_z = 122,
	SDLK_KP2 = 258, SDLK_KP4 = 260, SDLK_KP6 = 262, SDLK_KP8 = 264,
	SDLK_UP = 273, SDLK_DOWN = 274, SDLK_RIGHT = 275, SDLK_LEFT = 276,
	SDLK_RSHIFT = 303, SDLK_LSHIFT = 304
};

static inline SDL_Surface *fake_surface(int w, int h)
{
	static SDL_PixelFormat fmt;
	SDL_Surface *s = calloc(1, sizeof *s);
	s->format = &fmt;
	s->w = w; s->h = h; s->pitch = w;
	s->pixels = calloc((size_t)w * h, 1);
	return s;
}
static inline int SDL_Init(Uint32 f) { (void)f; return 0; }
static inline void SDL_Quit(void) {}
static inline const char *SDL_GetError(void) { return "fake"; }
static inline SDL_Surface *SDL_SetVideoMode(int w, int h, int bpp, Uint32 f) { (void)bpp; (void)f; return fake_surface(w, h); }
static inline SDL_Surface *SDL_CreateRGBSurface(Uint32 f, int w, int h, int d, Uint32 r, Uint32 g, Uint32 b, Uint32 a)
{ (void)f; (void)d; (void)r; (void)g; (void)b; (void)a; return fake_surface(w, h); }
static inline int SDL_SetColors(SDL_Surface *s, SDL_Color *c, int f, int n) { (void)s; (void)c; (void)f; (void)n; return 1; }
static inline int SDL_BlitSurface(SDL_Surface *a, SDL_Rect *r, SDL_Surface *b, SDL_Rect *r2) { (void)a; (void)r; (void)b; (void)r2; return 0; }
static inline int SDL_Flip(SDL_Surface *s) { (void)s; return 0; }
static inline Uint32 SDL_GetTicks(void) { return 0; }
static inline void SDL_Delay(Uint32 ms) { (void)ms; }
static inline int SDL_PollEvent(SDL_Event *e) { e->type = 0; return 0; }
extern unsigned char fake_keys[512];
static inline Uint8 *SDL_GetKeyState(int *n) { (void)n; return fake_keys; }
static inline void SDL_WM_SetCaption(const char *a, const char *b) { (void)a; (void)b; }
#endif
