#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>
using namespace thrust;


#include "ragridder_plan.h"
#include "conv_interp_invoker.h"
#include "cuft.h"
#include "cugridder.h"
#include "precomp.h"
#include "utils.h"

///conv improved WS, method 0 correctness cheak

int main(int argc, char *argv[])
{
	// suppose there is just one channel
	// range of uvw [-lamda/2,lamda/2]， rescale with factor resolution / fov compatible with l
	// l and m need to be converted into pixels
	/* Input: nrow, nchan, nxdirty, nydirty, fov, epsilon
		row - number of visibility
		nchan - number of channels
		nxdirty, nydirty - image size (height width)
		fov - field of view
		epsilon - tolerance
	*/
	int ier = 0;
	if (argc < 7)
	{
		fprintf(stderr,
				"Usage: W Stacking\n"
				"Arguments:\n"
				"  method: One of\n"
				"    0: nupts driven,\n"
				"    2: sub-problem, or\n"
				"    4: block gather (each nf must be multiple of 8).\n"
				"  w_term_method: \n"
				"    0: w-stacking\n"
				"    1: improved w-stacking\n"
				"  nxdirty, nydirty : image size.\n"
				"  nrow: The number of non-uniform points.\n"
				"  fov: Field of view.\n"
				"  nchan: number of chanels (default 1)\n"
				"  epsilon: NUFFT tolerance (default 1e-6).\n"
				"  kerevalmeth: Kernel evaluation method; one of\n"
				"     0: Exponential of square root (default), or\n"
				"     1: Horner evaluation.\n");
		return 1;
	}
	// issue related to accuary - how to set sigma, epsilon, number of plane, beta and kw. the number of w plane may need to increase.
	int nxdirty, nydirty;
	PCS sigma = 4; // upsampling factor
	int nrow, nchan;
	PCS fov;

	double inp;
	int method;
	sscanf(argv[1], "%d", &method);
	int w_term_method;
	sscanf(argv[2], "%d", &w_term_method);
	sscanf(argv[3], "%d", &nxdirty);
	sscanf(argv[4], "%d", &nydirty);
	sscanf(argv[5], "%d", &nrow);
	sscanf(argv[6], "%lf", &inp);
	fov = inp;

	nchan = 1;
	if (argc > 7)
	{
		sscanf(argv[7], "%d", &nchan);
	}

	PCS epsilon = 1e-6;
	if (argc > 8)
	{
		sscanf(argv[8], "%lf", &inp);
		epsilon = (PCS)inp; // so can read 1e6 right!
	}

	int kerevalmeth = 0;
	if (argc > 9)
	{
		sscanf(argv[9], "%d", &kerevalmeth);
	}

	// degree per pixel (unit radius)
	// PCS deg_per_pixelx = fov / 180.0 * PI / (PCS)nxdirty;
	// PCS deg_per_pixely = fov / 180.0 * PI / (PCS)nydirty;
	// chanel setting
	PCS f0 = 1e9;
	PCS *freq = (PCS *)malloc(sizeof(PCS) * nchan);
	for (int i = 0; i < nchan; i++)
	{
		freq[i] = f0 + i / (double)nchan * fov; //!
	}
	//improved WS stacking 1,
	//gpu_method == 0, nupts driven

	//N1 = 5; N2 = 5; M = 25; //for correctness checking
	//int ier;
	PCS *u, *v, *w;
	CPX *vis;
	PCS *wgt=NULL; //currently no mask
	u = (PCS *)malloc(nrow * sizeof(PCS)); //Allocates page-locked memory on the host.
	v = (PCS *)malloc(nrow * sizeof(PCS));
	w = (PCS *)malloc(nrow * sizeof(PCS));
	vis = (CPX *)malloc(nrow * sizeof(CPX));
	PCS *d_u, *d_v, *d_w;
	CUCPX *d_vis, *d_fk;
	checkCudaErrors(cudaMalloc((void**)&d_u, nrow * sizeof(PCS)));
	checkCudaErrors(cudaMalloc((void**)&d_v, nrow * sizeof(PCS)));
	checkCudaErrors(cudaMalloc((void**)&d_w, nrow * sizeof(PCS)));
	checkCudaErrors(cudaMalloc((void**)&d_vis, nrow * sizeof(CUCPX)));

	// generating data
	for (int i = 0; i < nrow; i++)
	{
		u[i] = randm11() *  PI; //xxxxx remove
		v[i] = randm11() *  PI;
		w[i] = randm11() *  PI;
		vis[i].real(randm11()*1000); // nrow vis per channel, weight?
		vis[i].imag(randm11()*1000);
		// wgt[i] = 1;
	}
#ifdef DEBUG
	printf("origial input data...\n");
	for(int i=0; i<nrow; i++){
		printf("%.3lf ",u[i]);
	}
	printf("\n");
	for(int i=0; i<nrow; i++){
		printf("%.3lf ",vis[i].real());
	}
	printf("\n");
#endif
	// ignore the tdirty
	// how to convert ms to vis

	//printf("generated data, x[1] %2.2g, y[1] %2.2g , z[1] %2.2g, c[1] %2.2g\n",x[1] , y[1], z[1], c[1].real());

	// Timing begin
	//data transfer
	checkCudaErrors(cudaMemcpy(d_u, u, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //u
	checkCudaErrors(cudaMemcpy(d_v, v, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //v
	checkCudaErrors(cudaMemcpy(d_w, w, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //w

	/* -----------Step1: Baseline setting--------------
	skip negative v
    uvw, nrow = M, shift, mask, f_over_c (fixed due to single channel)
    */
	
	// int mask = (int(1) << shift) - 1; // ???
	PCS *f_over_c = (PCS*) malloc(sizeof(PCS)*nchan);
	for(int i=0; i<nchan; i++){
		f_over_c[i] = freq[i]/SPEEDOFLIGHT;
	}

	/* ----------Step2: cugridder------------*/
	// plan setting
	curafft_plan *plan;

	ragridder_plan *gridder_plan;

	plan = new curafft_plan();
    gridder_plan = new ragridder_plan();
    memset(plan, 0, sizeof(curafft_plan));
    memset(gridder_plan, 0, sizeof(ragridder_plan));
	
	visibility *pointer_v;
	pointer_v = (visibility *)malloc(sizeof(visibility));
	pointer_v->u = u;
	pointer_v->v = v;
	pointer_v->w = w;
	pointer_v->vis = vis;
	pointer_v->frequency = freq;
	pointer_v->weight = wgt;
	pointer_v->pirange = 1;

	int direction = 1; //inverse

	// device data allocation and transfer should be done in gridder setting
	ier = gridder_setting(nydirty,nxdirty,method,kerevalmeth,w_term_method,epsilon,direction,sigma,0,1,nrow,nchan,fov,pointer_v,d_u,d_v,d_w,d_vis
		,plan,gridder_plan);
	//print the setting result
	free(pointer_v);



	if(ier == 1){
		printf("errors in gridder setting\n");
		return ier;
	}

    checkCudaErrors(cudaMalloc((void**)&plan->fw,sizeof(CUCPX)*plan->nf1*plan->nf2*plan->nf3));
    checkCudaErrors(cudaMemset(plan->fw, 0, plan->nf3 * plan->nf1 * plan->nf2 * sizeof(CUCPX)));
	// fk(image) malloc and set
	checkCudaErrors(cudaMalloc((void**)&d_fk,sizeof(CUCPX)*nydirty*nxdirty));
	plan->fk = d_fk;

	gridder_plan->dirty_image = (CPX *)malloc(sizeof(CPX)*nxdirty*nydirty*nchan); //

	// how to use weight flag and frequency
	for(int i=0; i<nchan; i++){
		// pre_setting
		// 1. u, v, w * f_over_c
		// 2. /pixelsize(*2pi)
		// 3. * rescale ratio
		// pre_setting(d_u, d_v, d_w, d_vis, plan, gridder_plan);
		// memory transfer (vis belong to this channel and weight)
		checkCudaErrors(cudaMemcpy(d_vis, vis, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice)); //
		// shift to corresponding range
		ier = gridder_execution(plan,gridder_plan);
		if(ier == 1){
			printf("errors in gridder execution\n");
			return ier;
		}
		checkCudaErrors(cudaMemcpy(gridder_plan->dirty_image+i*nxdirty*nydirty, d_fk, sizeof(CUCPX)*nydirty*nxdirty,
			cudaMemcpyDeviceToHost));
	}
	printf("result printing...\n");
	for(int i=0; i<nxdirty; i++){
		for(int j=0; j<nydirty; j++){
			printf("%.5lf ", gridder_plan->dirty_image[i*nydirty+j].real());
		}
		printf("\n");
	}

	PCS pi_ratio = 1;
	if(!gridder_plan->kv.pirange)pi_ratio = 2 * PI;

	printf("ground truth printing...\n");
	for(int i=0; i<nxdirty; i++){
		for(int j=0; j<nydirty; j++){
			CPX temp(0.0,0.0);
			PCS n_lm = sqrt(1.0-pow(gridder_plan->pixelsize_x*(i-nxdirty/2),2)-pow(gridder_plan->pixelsize_y*(j-nydirty/2),2));
			for(int k=0; k<nrow; k++){
				temp += vis[k]*exp((u[k]*pi_ratio*(i-nxdirty/2)+v[k]*pi_ratio*(j-nydirty/2)+w[k]*pi_ratio*(n_lm-1))*IMA);
			}
			printf("%lf ",temp.real()/(n_lm));
		}
		printf("\n");
	}
	
	ier = gridder_destroy(plan, gridder_plan);
	if(ier == 1){
		printf("errors in gridder destroy\n");
		return ier;
	}


	return ier;
}