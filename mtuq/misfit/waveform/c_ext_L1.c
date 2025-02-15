#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>

#define data_data(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(data_data)+\
    (i0) * PyArray_STRIDES(data_data)[0]+\
    (i1) * PyArray_STRIDES(data_data)[1])))

#define greens_data(i0,i1,i2,i3) \
    (*(npy_float64*)((PyArray_DATA(greens_data)+\
    (i0) * PyArray_STRIDES(greens_data)[0]+\
    (i1) * PyArray_STRIDES(greens_data)[1]+\
    (i2) * PyArray_STRIDES(greens_data)[2]+\
    (i3) * PyArray_STRIDES(greens_data)[3])))

#define greens_greens(i0,i1,i2,i3,i4) \
    (*(npy_float64*)((PyArray_DATA(greens_greens)+\
    (i0) * PyArray_STRIDES(greens_greens)[0]+\
    (i1) * PyArray_STRIDES(greens_greens)[1]+\
    (i2) * PyArray_STRIDES(greens_greens)[2]+\
    (i3) * PyArray_STRIDES(greens_greens)[3]+\
    (i4) * PyArray_STRIDES(greens_greens)[4])))

#define sources(i0,i1) \
    (*(npy_float64*)((PyArray_DATA(sources)+\
    (i0) * PyArray_STRIDES(sources)[0]+\
    (i1) * PyArray_STRIDES(sources)[1])))

#define groups(i0,i1) \
    (*(npy_float64*)((PyArray_DATA(groups)+\
    (i0) * PyArray_STRIDES(groups)[0]+\
    (i1) * PyArray_STRIDES(groups)[1])))

#define weights(i0,i1) \
    (*(npy_float64*)((PyArray_DATA(weights)+\
    (i0) * PyArray_STRIDES(weights)[0]+\
    (i1) * PyArray_STRIDES(weights)[1])))

#define results(i0) \
    (*(npy_float64*)((PyArray_DATA(results)+\
    (i0) * PyArray_STRIDES(results)[0])))

#define cc(i0) \
    (*(npy_float64*)((PyArray_DATA(cc)+\
    (i0) * PyArray_STRIDES(cc)[0])))

static PyObject *misfit(PyObject *self, PyObject *args) {

   PyArrayObject *data_data, *greens_data, *greens_greens;
   PyArrayObject *sources, *groups, *weights;

   int hybrid_norm;
   npy_float64 dt;
   int NPAD1, NPAD2;
   int debug_level;
   int msg_start, msg_stop, msg_percent;

   int NSRC, NSTA, NC, NG, NGRP;
   int isrc, ista, ic, ig, igrp;

   int cc_argmax, it, itpad, j1, j2, nd, NPAD;
   npy_float64 cc_max, L1_sum, L1_tmp;

   float iter, next_iter;
   int msg_count, msg_interval;

   if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!idiiiiii",
                         &PyArray_Type, &data_data,
                         &PyArray_Type, &greens_data,
                         &PyArray_Type, &greens_greens,
                         &PyArray_Type, &sources,
                         &PyArray_Type, &groups,
                         &PyArray_Type, &weights,
                         &hybrid_norm,
                         &dt,
                         &NPAD1,
                         &NPAD2,
                         &debug_level,
                         &msg_start,
                         &msg_stop,
                         &msg_percent)) {
    return NULL;
  }

  NSRC = (int) PyArray_SHAPE(sources)[0];
  NSTA = (int) PyArray_SHAPE(weights)[0];
  NC = (int) PyArray_SHAPE(weights)[1];
  NG = (int) PyArray_SHAPE(sources)[1];
  NGRP = (int) PyArray_SHAPE(groups)[0];

  NPAD = (int) NPAD1 + NPAD2 + 1;

  if (debug_level > 1) {
    printf(" number of sources:  %d\n", NSRC);
    printf(" number of stations:  %d\n", NSTA);
    printf(" number of components:  %d\n", NC);
    printf(" number of Green's functions:  %d\n\n", NG);
    printf(" number of component groups:  %d\n", NGRP);
  }

  nd = 1;
  npy_intp dims_cc[] = {(int) NPAD};
  PyObject *cc = PyArray_SimpleNew(nd, dims_cc, NPY_DOUBLE);

  nd = 2;
  npy_intp dims_results[] = {(int) NSRC, 1};
  PyObject *results = PyArray_SimpleNew(nd, dims_results, NPY_DOUBLE);

  if (msg_percent > 0) {
    msg_interval = msg_percent / 100. * msg_stop;
    msg_count = 100. / msg_percent * msg_start / msg_stop;
    iter = (float) msg_start;
    next_iter = (float) msg_count * msg_interval;
  } else {
    iter = 0;
    next_iter = INFINITY;
  }

  for (isrc = 0; isrc < NSRC; ++isrc) {

    if (iter >= next_iter) {
        printf("  about %d percent finished\n", msg_percent * msg_count);
        msg_count += 1;
        next_iter = msg_count * msg_interval;
    }
    iter += 1;

    L1_sum = (npy_float64) 0.;

    for (ista = 0; ista < NSTA; ista++) {
      for (igrp = 0; igrp < NGRP; igrp++) {

        for (it = 0; it < NPAD; it++) {
          cc(it) = (npy_float64) 0.;
        }

        for (ic = 0; ic < NC; ic++) {

          if (((int) groups(igrp, ic)) == 0) {
            continue;
          }

          if (fabs(weights(ista, ic)) < 1.e-6) {
            continue;
          }

          for (ig = 0; ig < NG; ig++) {
            for (it = 0; it < NPAD; it++) {
                cc(it) += greens_data(ista, ic, ig, it) * sources(isrc, ig);
            }
          }
        }
        cc_max = -NPY_INFINITY;
        cc_argmax = 0;
        for (it = 0; it < NPAD; it++) {
          if (cc(it) > cc_max) {
            cc_max = cc(it);
            cc_argmax = it;
          }
        }
        itpad = cc_argmax;

        for (ic = 0; ic < NC; ic++) {
          L1_tmp = 0.;

          if (((int) groups(igrp,ic))==0) {
            continue;
          }

          // Skip traces that have been assigned zero weight
          if (fabs(weights(ista, ic)) < 1.e-6) {
              continue;
          }

          // calculate s^2
          for (j1=0; j1<NG; j1++) {
            for (j2=0; j2<NG; j2++) {
              L1_tmp += sources(isrc, j1) * sources(isrc, j2) *
                  greens_greens(ista,ic,itpad,j1,j2);
            }
          }

          // calculate d^2
          L1_tmp += data_data(ista,ic);

          // calculate sd
          for (ig=0; ig<NG; ig++) {
            L1_tmp -= 2.*greens_data(ista,ic,ig,itpad) * sources(isrc, ig);
          }

          // for L1 norm
          L1_tmp = sqrt(L1_tmp);

          L1_sum += dt * weights(ista, ic) * L1_tmp;
        }

      }
    }
    results(isrc) = L1_sum;

  }
  return results;
}

static PyMethodDef methods[] = {
    { "misfit", misfit, METH_VARARGS, "Misfit function (L1 norm implementation)."},
    { NULL, NULL, 0, NULL }
  };

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef misfit_module = {
  PyModuleDef_HEAD_INIT,
  "c_ext_L1",
  "Misfit function (L1 norm implementation)",
  -1,                  /* m_size */
  methods,             /* m_methods */
  };
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_c_ext_L1(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&misfit_module);
  }
#else
PyMODINIT_FUNC initc_ext_L1(void) {
  (void) Py_InitModule("c_ext_L1", methods);
  import_array();
  }
#endif