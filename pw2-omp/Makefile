CXX = g++
CXXFLAGS = -O3 -fopenmp `pkg-config opencv4 --cflags`
LDFLAGS = -O3 -fopenmp `pkg-config opencv4 --libs` -lstdc++

TARGETS = ex1-1-anaglyph ex1-2-gaussian-filtering ex3-comparison ex1-3-denoising ex1-1-anaglyph ex2-shared-memory-optim anaglyphMethods

all: $(TARGETS)

opencv-omp-average-kernel-padding: opencv-omp-average-kernel-padding.o
	$(CXX) $< $(LDFLAGS) -o $@

ex1-1-anaglyph: ex1-1-anaglyph.o anaglyphMethods.o
	$(CXX) $^ $(LDFLAGS) -o $@ 

ex1-2-gaussian-filtering: ex1-2-gaussian-filtering.o
	$(CXX) $< $(LDFLAGS) -o $@

ex1-3-denoising: ex1-3-denoising.o
	$(CXX) $< $(LDFLAGS) -o $@

ex2-shared-memory-optim: ex2-shared-memory-optim.o
	$(CXX) $< $(LDFLAGS) -o $@

ex3-comparison: ex3-comparison.o
	$(CXX) $< $(LDFLAGS) -o $@

test: test.o
	$(CXX) $< $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<  -o $@

clean:
	rm -f *.o $(TARGETS)
	rm -f results/*.png
