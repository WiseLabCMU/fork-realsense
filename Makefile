program_NAME := fork_realsense
program_CXX_SRCS := $(wildcard src/*.cpp) 
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_CXX_FLAGS := -std=c++11 
program_OBJS := $(program_CXX_OBJS)
program_INCLUDE_DIRS := include/
program_LIBRARIES := boost_system boost_filesystem boost_date_time realsense pthread \
					 opencv_core opencv_imgproc opencv_highgui

CXXFLAGS += $(foreach cxxflag,$(program_CXX_FLAGS),$(cxxflag))
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))

.PHONY: all clean distclean

all: $(program_NAME)

$(program_NAME): $(program_OBJS)
	$(LINK.cc) $(program_OBJS) -o $(program_NAME) 

clean:
	@- $(RM) $(program_NAME)
	@- $(RM) $(program_OBJS)

distclean: clean