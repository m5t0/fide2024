# CMakeList.txt : fide2024 の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#

# ソースをこのプロジェクトの実行可能ファイルに追加します。
file(GLOB CPP_SOURCES "${PROJECT_SOURCE_DIR}/fide2024/*.cpp")
file(GLOB H_SOURCES "${PROJECT_SOURCE_DIR}/fide2024/*.h")

set(IS_KAGGLE_ENV False CACHE BOOL "Kaggle環境かどうかを表す変数")

# Kaggle環境の場合はビルドしない
if(NOT IS_KAGGLE_ENV)
  add_library (fide2024_lib ${CPP_SOURCES})
endif()

add_executable (fide2024 ${H_SOURCES} ${CPP_SOURCES})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET fide2024 PROPERTY CXX_STANDARD 20)
endif()

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
