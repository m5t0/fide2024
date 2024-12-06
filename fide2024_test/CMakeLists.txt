# CMakeList.txt : fide2024 の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#

# ソースをこのプロジェクトの実行可能ファイルに追加します。
# 単体テスト

include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

# Now simply link against gtest or gtest_main as needed.
# ソースをこのプロジェクトの実行可能ファイルに追加します。
file(GLOB CPP_SOURCES "${PROJECT_SOURCE_DIR}/fide2024_test/*.cpp")
add_executable (fide2024_test ${CPP_SOURCES})
include_directories(fide2024_test "${PROJECT_SOURCE_DIR}/fide2024")
target_link_libraries(fide2024_test gtest_main fide2024_lib)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET fide2024 PROPERTY CXX_STANDARD 20)
endif()

include(GoogleTest)
gtest_discover_tests(fide2024_test)

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
