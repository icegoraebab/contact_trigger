cmake_minimum_required(VERSION 3.8)
project(contact_trigger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── 의존성 ───────────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# ── 실행파일 ─────────────────────────────────────────────────────────────────
add_executable(reflex_grasp_node
  src/reflex_grasp_node.cpp
)

target_include_directories(reflex_grasp_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(reflex_grasp_node
  rclcpp
  std_msgs
  sensor_msgs
)

# ── 설치 ─────────────────────────────────────────────────────────────────────
install(TARGETS reflex_grasp_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()