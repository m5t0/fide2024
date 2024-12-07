# fide2024
This code written for the kaggle competition, ["FIDE & Google Efficient Chess AI Challenge"](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/overview).

## Environment
- Windows 11
- Visual Studio Community 2022 Version 17.12.3
- CMake Version 3.29(attached with Visual Studio Community 2022)
- Visual C++
- Google Test Version 1.12.1

## Folders and Files
- fide2024
	- fide2024.cpp
		- entry point
- fide2024_test
	- fide2024_test.cpp
		- test code for ...(this is a dummy file)

## Usage
I'll explain how to build on kaggle notebook.
Please download this repository and run the following commands to build the source code.

1. `cd ./(the path of this repository)`
1. `cmake -DIS_KAGGLE_ENV=True -DCMAKE_BUILD_TYPE="Release" -S /kaggle/input/fide2024-cpp-code/fide2024  -B build`
1. `!cmake --build build`

If you set `IS_KAGGLE_ENV=False`, projects that aren't needed for submitting, such as tests, will also be built.

The path of the execution file after building the source code is `(the path of this repository)/build/fide2024/fide2024`.
