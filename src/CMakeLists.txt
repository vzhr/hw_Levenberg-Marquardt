cmake_minimum_required(VERSION 3.10)
project(hw_lm VERSION 1.0.0 LANGUAGES CXX)

string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(NAME "hw_lm")
add_library(${NAME} SHARED  ${CMAKE_CURRENT_SOURCE_DIR}/edge.cpp)
target_sources(${NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/edge.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/edge.h
        ${CMAKE_CURRENT_SOURCE_DIR}/eigen_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/macros.h
        ${CMAKE_CURRENT_SOURCE_DIR}/misc.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer_lm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer_lm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/se3_ops.h
        ${CMAKE_CURRENT_SOURCE_DIR}/se3_ops.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/se3quat.h
        ${CMAKE_CURRENT_SOURCE_DIR}/types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vertex.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vertex.cpp

        )
target_compile_options(${NAME} PRIVATE -fPIC
        -Wpedantic -Werror=reorder -Werror=return-type)
if (BUILD_TYPE STREQUAL "DEBUG")
    target_compile_options(${NAME} PRIVATE -g)
    target_compile_definitions(${NAME} PRIVATE DEBUG)
else ()
    target_compile_options(${NAME} PRIVATE -O3)
endif ()
target_compile_features(${NAME} PRIVATE cxx_std_11)

find_package(Eigen3 REQUIRED)
target_include_directories(${NAME}
        INTERFACE

        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:include/${NAME}>
        PRIVATE


        )

target_link_libraries(${NAME}
        PRIVATE
        Eigen3::Eigen
        )


# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Use:
#   * PROJECT_VERSION
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "cmake/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${generated_dir}"
)



install(TARGETS ${NAME}
        EXPORT ${TARGETS_EXPORT_NAME}
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

#install(
#    DIRECTORY ""
#    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${NAME}"
#    FILES_MATCHING PATTERN "*.h"
#)
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${NAME}")


# Config
install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)
install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)

export(EXPORT ${TARGETS_EXPORT_NAME}
        FILE cmake/${TARGETS_EXPORT_NAME}_build.cmake)
