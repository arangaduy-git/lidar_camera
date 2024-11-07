/* File: rplidar.i */
/* File: rplidar.i */
%module rplidar


%{
    #define SWIG_FILE_WITH_INIT
    #include <vector>
    #include "rplidar.hpp"
%}

%include <std_vector.i>

/*%array_class(double, doubleArray)*/

%typemap(out) std::vector<double>& {
   PyObject* pylist = PyList_New($1->size());
   int i = 0;
   for (const auto &val: *$1) {
      PyObject* pydouble = PyFloat_FromDouble(val);
      PyList_SET_ITEM(pylist, i, pydouble);
      i++;
   }
   $result = pylist;
}

%include "rplidar.hpp"