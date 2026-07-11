/* Original C++ source for OptArena kernel tsvc_2_va. Upstream: Vectra Artifacts (Work/VectraArtifacts) tsvc microkernels. Timing instrumentation removed. License: see upstream. Not the scoring oracle -- the numpy reference remains the correctness oracle. */

#include <cstdint>
#include <cmath>

extern "C" {

// -----------------------------------------------------------------------------
// %5.1  va_d
// -----------------------------------------------------------------------------
void va_d(double *__restrict__ a, const double *__restrict__ b,
                  int iterations, int len_1d) {

  
    for (int i = 0; i < len_1d; ++i) {
      a[i] = b[i];
    }
  

}

} // extern "C"
