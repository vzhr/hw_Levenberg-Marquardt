cmake_minimum_required(VERSION 3.10)
project(hw_lm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(NAME "hw_lm")
add_library(${NAME} SHARED )
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
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:${INCLUDES_DESTINATION}/${NAME}>
        ${EIGEN3_INCLUDE_DIRS}
        PRIVATE
        )

target_link_libraries(${NAME}
        PRIVATE
        )

set(EXPORTNAME ${NAME}Targets)
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${NAME}Config.cmake @ONLY)

install(TARGETS ${NAME}
        EXPORT ${EXPORTNAME}
        RUNTIME DESTINATION ${RUNTIME_DESTINATION}
        LIBRARY DESTINATION ${LIBRARY_DESTINATION}
        ARCHIVE DESTINATION ${ARCHIVE_DESTINATION})
install(EXPORT ${EXPORTNAME}
        #        FILE ${EXPORTNAME}.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${LIBRARY_DESTINATION}/cmake/${NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${NAME}Config.cmake
        DESTINATION ${LIBRARY_DESTINATION}/cmake/${NAME})

#install(DIRECTORY include/
#        DESTINATION ${INCLUDES_DESTINATION})
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${INCLUDES_DESTINATION}/${NAME})

export(EXPORT ${EXPORTNAME}
        FILE cmake/${EXPORTNAME}_build.cmake)
