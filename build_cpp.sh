export PYTHONPATH=$PYTHONPATH:`readlink -f python`:`readlink -f cpp/build`
cd cpp/
cd build
cmake -Dpybind11_DIR=/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11 -DPython_EXECUTABLE=/usr/bin/python3 ..
make -j
cd ../..
