# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/sequential_histogram_equalization.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sequential_histogram_equalization.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sequential_histogram_equalization.dir/flags.make

CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o: CMakeFiles/sequential_histogram_equalization.dir/flags.make
CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o"
	/usr/local/Cellar/llvm/6.0.0/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o -c "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/main.cpp"

CMakeFiles/sequential_histogram_equalization.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sequential_histogram_equalization.dir/main.cpp.i"
	/usr/local/Cellar/llvm/6.0.0/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/main.cpp" > CMakeFiles/sequential_histogram_equalization.dir/main.cpp.i

CMakeFiles/sequential_histogram_equalization.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sequential_histogram_equalization.dir/main.cpp.s"
	/usr/local/Cellar/llvm/6.0.0/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/main.cpp" -o CMakeFiles/sequential_histogram_equalization.dir/main.cpp.s

CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.requires

CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.provides: CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/sequential_histogram_equalization.dir/build.make CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.provides

CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.provides.build: CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o


# Object files for target sequential_histogram_equalization
sequential_histogram_equalization_OBJECTS = \
"CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o"

# External object files for target sequential_histogram_equalization
sequential_histogram_equalization_EXTERNAL_OBJECTS =

sequential_histogram_equalization: CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o
sequential_histogram_equalization: CMakeFiles/sequential_histogram_equalization.dir/build.make
sequential_histogram_equalization: /usr/local/lib/libopencv_stitching.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_superres.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_videostab.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_aruco.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_bgsegm.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_bioinspired.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_ccalib.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_dnn_objdetect.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_dpm.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_face.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_fuzzy.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_hfs.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_img_hash.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_line_descriptor.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_optflow.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_reg.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_rgbd.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_saliency.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_stereo.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_structured_light.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_surface_matching.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_tracking.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_xfeatures2d.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_ximgproc.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_xobjdetect.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_xphoto.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_shape.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_photo.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_dnn.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_datasets.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_ml.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_plot.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_video.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_calib3d.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_features2d.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_highgui.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_videoio.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_phase_unwrapping.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_flann.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_imgcodecs.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_objdetect.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_imgproc.3.4.1.dylib
sequential_histogram_equalization: /usr/local/lib/libopencv_core.3.4.1.dylib
sequential_histogram_equalization: CMakeFiles/sequential_histogram_equalization.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sequential_histogram_equalization"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sequential_histogram_equalization.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sequential_histogram_equalization.dir/build: sequential_histogram_equalization

.PHONY : CMakeFiles/sequential_histogram_equalization.dir/build

CMakeFiles/sequential_histogram_equalization.dir/requires: CMakeFiles/sequential_histogram_equalization.dir/main.cpp.o.requires

.PHONY : CMakeFiles/sequential_histogram_equalization.dir/requires

CMakeFiles/sequential_histogram_equalization.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sequential_histogram_equalization.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sequential_histogram_equalization.dir/clean

CMakeFiles/sequential_histogram_equalization.dir/depend:
	cd "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization" "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization" "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/cmake-build-debug" "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/cmake-build-debug" "/Users/francesco/Documents/Università/magistrale/esami/Parallel Computing/Parallel-Histogram-Equalization/sequential_histogram_equalization/cmake-build-debug/CMakeFiles/sequential_histogram_equalization.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/sequential_histogram_equalization.dir/depend

