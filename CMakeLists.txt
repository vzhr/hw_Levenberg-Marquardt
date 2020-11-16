cmake_minimum_required(VERSION 3.10)

project(hw_LM_test VERSION 1.0.0 LANGUAGES CXX)



string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
set(NAME ${PROJECT_NAME})
add_executable(${NAME}
        main.cpp)
add_subdirectory(src)
target_compile_options(${PROJECT_NAME} PRIVATE -fPIC -Wno-dev
       -Wpedantic -Werror=reorder -Werror=return-type -Werror=return-type)

if (BUILD_TYPE STREQUAL "DEBUG")
    target_compile_options(${NAME} PRIVATE -g)
    target_compile_definitions(${NAME} PRIVATE DEBUG)
else ()
    target_compile_options(${NAME} PRIVATE -O3)
endif ()

find_package(Eigen3 REQUIRED)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        hw_lm
        )
