# Builds libspout.so for Python (ctypes).
SPOUT_SRC ?= ../spout
CFLAGS ?= -O3 -march=native -fPIC -Wall -Wextra -Werror -std=gnu11 -fopenmp
CPPFLAGS += -I. -I$(SPOUT_SRC)

all: libspout.so

libspout.so: spout_core.c spout_vec.c spout_core.h spout_vec.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -shared -o $@ spout_core.c spout_vec.c -lm

clean:
	rm -f libspout.so

.PHONY: all clean
