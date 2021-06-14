#ifndef __PRECOMP__H__
#define __PRECOMP__H__

#include <cuda.h>
#include <helper_cuda.h>
#include "dataType.h"
#include "ragridder_plan.h"

__global__ void gridder_rescaling(CUCPX *x, PCS scale_ratio, int N);
void explicit_gridder_invoker(ragridder_plan *gridder_plan);
void pre_setting(PCS *d_u, PCS *d_v, PCS *d_w, PCS d_vis, ragridder_plan *gridder_plan);

#endif