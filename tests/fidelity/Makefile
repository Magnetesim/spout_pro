# Runs the original spout.c and the port side by side and compares frames.
ROOT = ../..
SPOUT = $(ROOT)/spout
CSRC = $(ROOT)/csrc
BUILD = build

all: run

$(BUILD)/spout_patched.c: $(SPOUT)/spout.c
	mkdir -p $(BUILD)
	sed -e 's/static int gx\[\] = {-2, 2, -1, 1, 0};/\n#define gx test_gx\n/' \
	    -e 's/static int gPhase = 0;/\n#define gPhase test_gPhase\n/' $< > $@
	grep -q '#define gx test_gx' $@ && grep -q '#define gPhase test_gPhase' $@
	cp $(SPOUT)/config.def.h $(BUILD)/config.h

$(BUILD)/harness: harness.c $(BUILD)/spout_patched.c $(CSRC)/spout_core.c $(CSRC)/spout_core.h
	$(CC) -O2 -g -std=gnu11 -w -Ifake_sdl -I$(BUILD) -I$(SPOUT) -I$(CSRC) \
	    -o $@ harness.c $(CSRC)/spout_core.c -lm

run: $(BUILD)/harness
	./$(BUILD)/harness $(EPISODES)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
