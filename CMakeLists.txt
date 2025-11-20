cmake_minimum_required(VERSION 3.10)
project(Project04_CameraCalib)

# OpenCV
set(OPENCV_DIR "D:/Software_Installers/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tạo 1 target cho MỖI file main trong src/apps/*.cpp
file(GLOB APP_SRCS CONFIGURE_DEPENDS "src/apps/*.cpp")

foreach(app ${APP_SRCS})
    get_filename_component(app_name ${app} NAME_WE)
    add_executable(${app_name} ${app} ${COMMON_SRCS})
    target_link_libraries(${app_name} PRIVATE ${OpenCV_LIBS})
    target_include_directories(${app_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
    set_target_properties(${app_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()
