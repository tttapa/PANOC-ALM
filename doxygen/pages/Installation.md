# Installation instructions {#installation}

## Linux

### Tools
First, install a compiler, GNU Make, CMake, Git ...:
```sh
sudo apt install build-essential cmake git
```

### Clone the repository

```sh
git clone git@github.com:tttapa/PANOC-ALM.git
```
For the purposes of these instructions, we'll refer to the PANOC-ALM repository 
as the environment variable `PANOC_ALM_ROOT`, for example:
```sh
export PANOC_ALM_ROOT="$HOME/GitHub/PANOC-ALM"
```

### Create a virtual environment

```sh
cd "$PANOC_ALM_ROOT"
python3 -m venv py-venv
. ./py-venv/bin/activate
```

### Install dependencies

The `scripts` folder contains some Bash scripts to install the necessary 
dependencies. By default, these scripts install everything into the virtual
environment, they require no root privileges, and won't change any other parts
of your system. If you have some of these installed globally already (e.g. BLAS)
you can skip these scripts.

```sh
pip install -r scripts/requirements.txt
./scripts/install-openblas.sh   # https://www.openblas.net/
./scripts/install-eigen.sh      # https://eigen.tuxfamily.org/index.php
./scripts/install-casadi.sh     # https://web.casadi.org/
./scripts/install-cutest.sh     # https://github.com/ralna/CUTEst
./scripts/install-lbfgspp.sh    # https://github.com/yixuan/LBFGSpp
./scripts/install-yaml-cpp.sh   # https://github.com/jbeder/yaml-cpp
./scripts/install-gtest.sh      # https://google.github.io/googletest/
```

### Build and install

```sh
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
make install -j$(nproc) # Build and install the optimized release version
make test # Run the tests
cmake .. -DCMAKE_BUILD_TYPE=Debug
make install -j$(nproc) # Build and install the debug version
make test # Run the tests in debug mode (with extra checks)
```
Installing both the release and debug versions can be very useful for checking
matrix dimension errors and out of bounds accesses during development, and 
switching to an optimized version later.

If you install the library locally, as demonstrated in the previous snippet,
you might have to set some environment variables, as explained here:
https://tttapa.github.io/Pages/Ubuntu/Software-Installation/Installing-Locally.html

Specifically, you need to add `~/.local` to the `CMAKE_PREFIX_PATH` by adding
the following to your `~/.profile` file:
```sh
CMAKE_PREFIX_PATH="$HOME/.local:$CMAKE_PREFIX_PATH"
```
Then source it (`. ~/.profile`) or log out and back in again.

## Windows

==TODO==

## OSX

==TODO==

***

# Usage

Once the library is installed, you can use it in your own projects.

For example:

**main.cpp**
```cpp
#include <panoc-alm/alm.hpp>
#include <panoc-alm/inner/panoc.hpp>

int main() {
    // Use the solvers as shown in the examples
}
```

**CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.16)
project(Project)

# Find the library you just installed:
find_package(PANOC-ALM)

add_executable(main main.cpp)
# Link your executable with the library:
target_link_libraries(main PRIVATE panoc-alm::panoc-alm)
```

# Python

The Python module can be installed using:
```sh
python setup.py install
```