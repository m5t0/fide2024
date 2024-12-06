# CMakeList.txt : 最上位の CMake プロジェクト ファイル。グローバル構成を行います
# また、サブプロジェクトをここに含めます。
#
cmake_minimum_required (VERSION 3.8)

# サポートされている場合は、MSVC コンパイラのホット リロードを有効にします。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(IS_KAGGLE_ENV False CACHE BOOL "Kaggle環境かどうかを表す変数")

project ("fide2024")

# Kaggle環境の場合はビルドしない
if(NOT IS_KAGGLE_ENV)
  project ("fide2024_lib")
  project ("fide2024_test")
endif()

# サブプロジェクトを含めます。
add_subdirectory ("fide2024")

# Kaggle環境の場合はフォルダを含めない
if(NOT IS_KAGGLE_ENV)
  add_subdirectory ("fide2024_test")
endif()