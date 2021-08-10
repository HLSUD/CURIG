#ifndef __CONV_INTERP_INVOKER_H__
#define __CONV_INTERP_INVOKER_H__


#include "curafft_plan.h"
#include "conv.h"

#define CONV_THREAD_NUM 32

int setup_conv_opts(conv_opts &c_opts, PCS eps, PCS upsampfac, int pirange, int direction, int kerevalmeth);//cautious the &
int get_num_cells(int ms, conv_opts copts);
int curafft_conv(curafft_plan *plan);
int curafft_free(curafft_plan *plan);
int curafft_interp(curafft_plan * plan);
#endif
