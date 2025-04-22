CXX = g++
CXXFLAGS = -O3 -fopenmp `pkg-config opencv4 --cflags`
LDFLAGS = -O3 -fopenmp `pkg-config opencv4 --libs` -lstdc++

TARGETS = ex1-1-anaglyph ex1-2-gaussian ex1-3-denoising 

all: $(TARGETS)

ex1-1-anaglyph: ex1-1-anaglyph.o
	$(CXX) $^ $(LDFLAGS) -o $@ 

ex1-2-gaussian: ex1-2-gaussian.o
	$(CXX) $< $(LDFLAGS) -o $@

ex1-3-denoising: ex1-3-denoising.o
	$(CXX) $< $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<  -o $@

clean:
	rm -f *.o $(TARGETS)
	rm -f results/*.png
