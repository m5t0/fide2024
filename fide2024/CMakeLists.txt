# CMakeList.txt : fide2024 の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#

# ソースをこのプロジェクトの実行可能ファイルに追加します。
file(GLOB CPP_SOURCES "${PROJECT_SOURCE_DIR}/fide2024/*.cpp")
file(GLOB H_SOURCES "${PROJECT_SOURCE_DIR}/fide2024/*.h")
# file(GLOB CPP_SOURCES_SYZYGY "${PROJECT_SOURCE_DIR}/fide2024/syzygy/*.cpp")
# file(GLOB H_SOURCES_SYZYGY "${PROJECT_SOURCE_DIR}/fide2024/syzygy/*.h")

add_library(fide2024_lib ${H_SOURCES} ${CPP_SOURCES} ${H_SOURCES_SYZYGY} ${CPP_SOURCES_SYZYGY})
add_executable(fide2024 ${H_SOURCES} ${CPP_SOURCES} ${H_SOURCES_SYZYGY} ${CPP_SOURCES_SYZYGY})
# target_link_libraries(fide2024 pthread)
# target_link_libraries(fide2024_lib pthread)

target_compile_definitions(fide2024 PRIVATE USE_POPCNT=1)
target_compile_definitions(fide2024_lib PRIVATE USE_POPCNT=1)

if(IS_KAGGLE_ENV)
  target_compile_definitions(fide2024 PRIVATE KAGGLE=1)
  target_compile_definitions(fide2024_lib PRIVATE KAGGLE=1)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET fide2024 PROPERTY CXX_STANDARD 20)
endif()

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
