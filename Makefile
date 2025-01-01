# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -O3 -g -I./include -I/usr/local/include/opencv4
CUDAFLAGS = -I./include -I/usr/local/include/opencv4 -arch=sm_75
LDFLAGS = -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lcudart

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files and object files
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CXX_OBJECTS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
EXEC = $(BIN_DIR)/cudascripts

# Default target
all: $(EXEC)

# Linking
$(EXEC): $(CXX_OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(CXXFLAGS) $(CXX_OBJECTS) $(CUDA_OBJECTS) -o $(EXEC) $(LDFLAGS)

# Compiling C++ files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compiling CUDA files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(BUILD_DIR)/*.o $(EXEC)

# Phony targets
.PHONY: all clean

