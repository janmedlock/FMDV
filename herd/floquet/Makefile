# Build the Cython monodromy solver library.

NAME := monodromy
EXT += $(shell python3-config --extension-suffix)
MODULE := $(NAME)$(EXT)

CPPFLAGS := -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

CFLAGS := $(shell python3-config --cflags)
CFLAGS += -fPIC

LDFLAGS := $(shell python3-config --ldflags)
LDFLAGS += -lblas


default: $(MODULE)

%.c:	%.pyx
	cython3 $< -o $@

%$(EXT):	%.o
	$(CC) $(LDFLAGS) -shared $^ $(LDLIBS) -o $@

clean:
	$(RM) $(MODULE)
