include config.h

all: target

target: test

test: svm.o
	$(LINK) -o $@ $+ $(LDFLAGS) $(LIBRARIES)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $<

.c.o:
	$(CC) -c $(CFLAGS) $(INCLUDES) $<

%.o: %.cu
	$(NVCC) -c $(CUDA_FLAGS) $<

.PHONY: clean
clean:
	-rm -f *.o
	-rm -f test
