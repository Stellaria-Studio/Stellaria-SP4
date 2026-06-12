cmake_minimum_required(VERSION 3.20)
project(StellariaSP4SDK VERSION 0.1.0 LANGUAGES CXX)

option(SP4_BUILD_TOOLS "Build SP4 SDK command line tools" ON)

add_library(sp4sdk STATIC
  sdk/src/SP4Asset.cpp
  sdk/src/SP4Compiler.cpp
  sdk/src/SP4ECompiler.cpp
  sdk/src/SP4ERuntime.cpp
  sdk/src/SP4Runtime.cpp
  sdk/src/SafeTensorsReader.cpp
)

target_include_directories(sp4sdk PUBLIC sdk/include)
target_compile_features(sp4sdk PUBLIC cxx_std_17)

if(SP4_BUILD_TOOLS)
  add_executable(sp4c sdk/tools/sp4c.cpp)
  target_link_libraries(sp4c PRIVATE sp4sdk)

  add_executable(sp4inspect-cpp sdk/tools/sp4inspect.cpp)
  target_link_libraries(sp4inspect-cpp PRIVATE sp4sdk)

  add_executable(sp4ec sdk/tools/sp4ec.cpp)
  target_link_libraries(sp4ec PRIVATE sp4sdk)

  add_executable(sp4einspect-cpp sdk/tools/sp4einspect.cpp)
  target_link_libraries(sp4einspect-cpp PRIVATE sp4sdk)

  add_executable(sp4quantbench sdk/tools/sp4quantbench.cpp)
  target_link_libraries(sp4quantbench PRIVATE sp4sdk)

  if(APPLE)
    enable_language(OBJCXX)
    add_executable(sp4metal-validate sdk/tools/sp4metal_validate.mm)
    target_link_libraries(sp4metal-validate PRIVATE sp4sdk "-framework Foundation" "-framework Metal" "-framework QuartzCore")
    target_compile_features(sp4metal-validate PRIVATE cxx_std_17)
  endif()
endif()
