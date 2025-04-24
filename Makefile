# Makefile

# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

# Source files
SRCS = DP_NN.cpp DataLoader.cpp DPNoise.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Output binary
TARGET = dp_nn

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)
