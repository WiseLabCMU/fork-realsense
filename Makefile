CC = g++
CFLAGS = -std=c++11 `pkg-config opencv --cflags`
LDLIBS = -lrealsense -lpthread `pkg-config opencv --libs`
LDFLAGS = -L/lib64/

all: fork_imp

fork_imp: FORK_RS.cpp
	$(CC) $(CFLAGS) $? $(LDLIBS) $(LDFLAGS) -o $@

clean:
	rm fork_imp