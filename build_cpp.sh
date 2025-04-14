export PYTHONPATH=$PYTHONPATH:`readlink -f python`:`readlink -f cpp/build`
git submodule update --init --recursive
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cd cpp/
if [ -d "build" ]; then rm -rf build; fi
mkdir build
cd build

#PYBIND_DIR=$(python3 -c "import os.path, pybind11; print(os.path.join(os.path.split(pybind11.__file__)[0], 'share/cmake/pybind11'))")
#PYTHON_EXEC=$(python3 -c "import sys; print(sys.executable)")
#CMAKE_CMD="cmake -Dpybind11_DIR=$PYBIND_DIR -DPython_EXECUTABLE=$PYTHON_EXEC .."
#echo "Running: $CMAKE_CMD"
#eval $CMAKE_CMD

cmake -Dpybind11_DIR=/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11 -DPython_EXECUTABLE=/usr/bin/python3 ..
make -j
cd ../..


