# Makefile

# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

# Source files
SRCS = NN.cpp DataLoader.cpp 

# Object files
OBJS = $(SRCS:.cpp=.o)

# Output binary
TARGET = nn

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
