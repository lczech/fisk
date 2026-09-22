# This makefile wraps cmake so that a plain `make [-jN]` builds everything. Relies on the cmake
# "Unix Makefiles" generator; switching to Ninja would need a different mechanism to forward -j.

all: build/CMakeCache.txt build
.PHONY: all

# Run cmake if not yet done, or if CMakeLists.txt has changed.
build/CMakeCache.txt: CMakeLists.txt
	@mkdir -p build
	@cd build && cmake ..

# CONFIGURE_DEPENDS (see CMakeLists.txt) makes cmake re-glob and reconfigure on its own whenever
# a source file is added or removed, so this needs no separate "touch everything" step.
build: build/CMakeCache.txt
	$(MAKE) -C build
.PHONY: build

clean:
	rm -rf build bin
.PHONY: clean
