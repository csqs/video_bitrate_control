### Makefile for building tensorflow application
# link pre-build libtensorflow_cc.so to your personal build project
# param:
# INCLUDES: -I, remember to include eigen3 and tf libs
# LDLIBS : -L path of folder where libtensorflow_cc.so exist
# ACTUAL_LIBS: -l   name of actual lib(name).so

# g++ -std=c++11 -Wl,-rpath='$ORIGIN/lib' -Iinclude -Llib test.cpp -ltensorflow_cc -o exec

SOURCE_DIR =.
BIN_DIR = .

# CPP  = g++ -std=c++11
# CPP     = g++ -std=c++11 -Wl,-rpath='$ORIGIN/lib'
CPP     = g++ -g -D_DEBUG -std=c++11 -O2 -march=native -Wl,-rpath='lib'
# LDFLAGS = -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w -D_GLIBCXX_USE_CXX11_ABI=1
# LDFLAGS = 
# LDLIBS  = -L/usr/local/lib/
LDLIBS  = -Llib
INCLUDES = -Iinclude
ACTUAL_LIBS = -ltensorflow_cc 

INPUT_FILE = main.cpp \
			 client.cpp \
			 server.cpp \
			 datastruct.cpp \
			 variable.cpp \
			 include/a3cmodel.cpp\
			 tools/easylogging++.cc\

OBJET_FILE = RC_A3C_IPC
# $(CPP)  $(INCLUDES) $(LDFLAGS) $(LDLIBS)  $(INPUT_FILE) $(ACTUAL_LIBS) -o $(OBJET_FILE)
# g++ -std=c++11 -Wl,-rpath='$ORIGIN/lib' -I./include -L./lib test.cpp-ltensorflow_cc   -o mainc
# 

$(OBJET_FILE):main.cpp
	$(CPP)  $(INCLUDES) $(LDFLAGS) $(LDLIBS) $(INPUT_FILE) $(OBJS) $(ACTUAL_LIBS) -o $(OBJET_FILE) -lrt -lpthread $(LDFLAGS)

clean:
	rm -f $(OBJET_FILE)
	find . -name "*.o"  | xargs rm -rf
