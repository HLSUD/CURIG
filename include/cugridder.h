#ifndef __CUGRIDDER_H__
#define __CUGRIDDER_H__

#include "curafft_plan.h"
#include "ragridder_plan.h"
#include "utils.h"

int gridder_setting(int N1, int N2, int method, int kerevalmeth, int w_term_method, PCS tol, int direction, double sigma, int iflag,
    int batchsize, int M, int channel, PCS fov, visibility *pointer_v, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan, ragridder_plan *gridder_plan);
int gridder_execution(curafft_plan *plan, ragridder_plan *gridder_plan);
int gridder_destroy(curafft_plan *plan, ragridder_plan *gridder_plan);
extern "C"{
int ms2dirty_2(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, PCS *wgt,  CPX *dirty, PCS epsilon, PCS sigma);
int ms2dirty_1(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, CPX *dirty, PCS epsilon, PCS sigma);
int dirty2ms_1(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, CPX *dirty, PCS epsilon, PCS sigma);
}
#endif