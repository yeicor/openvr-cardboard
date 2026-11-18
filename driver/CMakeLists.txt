cmake_minimum_required(VERSION 3.16)
project(openvrcardboard_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
  set(OVRC_PLATFORM_DIR "win64")
elseif(APPLE)
  set(OVRC_PLATFORM_DIR "osx64")
else()
  set(OVRC_PLATFORM_DIR "linux64")
endif()

set(DRIVER_NAME driver_openvrcardboard)
set(DRIVER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/openvrcardboard/bin/${OVRC_PLATFORM_DIR}")

add_library(${DRIVER_NAME} SHARED
  src/controller_device.cpp
  src/device_base.cpp
  src/device_provider.cpp
  src/driver_link.cpp
  src/hmd_device.cpp
  src/hmd_driver_factory.cpp
  src/logging.cpp
  src/settings.cpp
  src/time_util.cpp
  src/tracker_device.cpp
  src/udp_socket.cpp
)

target_include_directories(${DRIVER_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(${DRIVER_NAME} SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openvr")

set_target_properties(${DRIVER_NAME} PROPERTIES
  PREFIX ""
  OUTPUT_NAME "driver_openvrcardboard"
  LIBRARY_OUTPUT_DIRECTORY "${DRIVER_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${DRIVER_OUTPUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

# Multi-config generators (Visual Studio, Xcode) otherwise append a per-config
# subdirectory, which would break SteamVR's bin/<platform>/ layout.
foreach(OVRC_CONFIG Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${OVRC_CONFIG}" OVRC_CONFIG_UPPER)
  set_target_properties(${DRIVER_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${OVRC_CONFIG_UPPER} "${DRIVER_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${OVRC_CONFIG_UPPER} "${DRIVER_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${OVRC_CONFIG_UPPER} "${CMAKE_BINARY_DIR}/lib"
  )
endforeach()

if(WIN32)
  target_compile_definitions(${DRIVER_NAME} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  target_link_libraries(${DRIVER_NAME} PRIVATE ws2_32)
  if(MSVC)
    target_compile_options(${DRIVER_NAME} PRIVATE /W4 /permissive-)
  endif()
else()
  find_package(Threads REQUIRED)
  target_link_libraries(${DRIVER_NAME} PRIVATE Threads::Threads)
  target_compile_options(${DRIVER_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Copy the SteamVR driver resources next to the binary so the build directory
# can be copied or registered directly.
add_custom_command(TARGET ${DRIVER_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/openvrcardboard"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/driver.vrdrivermanifest"
          "${CMAKE_BINARY_DIR}/openvrcardboard/driver.vrdrivermanifest"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/resources"
          "${CMAKE_BINARY_DIR}/openvrcardboard/resources"
  COMMENT "Assembling SteamVR driver folder"
)

option(OVRC_BUILD_TESTS "Build the driver protocol self-test" ON)
if(OVRC_BUILD_TESTS)
  enable_testing()
  add_executable(ovrc_protocol_test tests/protocol_test.cpp)
  target_include_directories(ovrc_protocol_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(ovrc_protocol_test SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openvr")
  if(WIN32)
    target_compile_definitions(ovrc_protocol_test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(ovrc_protocol_test PRIVATE ws2_32)
  else()
    find_package(Threads REQUIRED)
    target_link_libraries(ovrc_protocol_test PRIVATE Threads::Threads)
    target_compile_options(ovrc_protocol_test PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME protocol_roundtrip COMMAND ovrc_protocol_test)
endif()

install(TARGETS ${DRIVER_NAME} LIBRARY DESTINATION "bin/${OVRC_PLATFORM_DIR}" RUNTIME DESTINATION "bin/${OVRC_PLATFORM_DIR}")
install(FILES driver.vrdrivermanifest DESTINATION ".")
install(DIRECTORY resources DESTINATION ".")
