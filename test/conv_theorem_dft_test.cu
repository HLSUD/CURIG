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
#include "deconv.h"
#include "cugridder.h"
#include "precomp.h"
#include "utils.h"


int main(int argc, char *argv[])
{
	/* Input: M, N1, N2, epsilon method
		method - conv method
		M - number of randomly distributed points
		N1, N2 - output size
		epsilon - tolerance
	*/
	int ier = 0;
	int N = 100;
	PCS sigma = 2.0; // upsampling factor // for not on grid points needs larger upsampling factor
	int M = 100;

	
	PCS epsilon = 1e-7;
	
	int kerevalmeth = 0;
	
	int method=0;

	//gpu_method == 0, nupts driven

	//int ier;
	PCS *u;
	CPX *c;
	u = (PCS *)malloc(M  * sizeof(PCS)); //Allocates page-locked memory on the host.
	c = (CPX *)malloc(M  * sizeof(CPX));
	PCS *d_u;
	CUCPX *d_c, *d_fk;
	CUCPX *d_fw;
	checkCudaErrors(cudaMalloc(&d_u, M  * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_c, M  * sizeof(CUCPX)));
    /// pixel size 
	// generating data
	for (int i = 0; i < M; i++)
	{
		u[i] = randm11()*PI; //xxxxx
		c[i].real(randm11()); // M vis per channel, weight?
		c[i].imag(randm11());
		// wgt[i] = 1;
	}

	PCS *k = (PCS*) malloc(sizeof(PCS)*N*10);
	// PCS pixelsize = 0.01;
	for (int i = 0; i < N; i++)
	{
		/* code */
		 k[i] = i-N/2;
		// k[i] = -abs(randm11());
		// k[i] = i/(double)N;
	}
	
	
	//data transfer
	checkCudaErrors(cudaMemcpy(d_u, u, M * sizeof(PCS), cudaMemcpyHostToDevice)); //u
	checkCudaErrors(cudaMemcpy(d_c, c, M * sizeof(CUCPX), cudaMemcpyHostToDevice));

	/* ----------Step2: plan setting------------*/
	curafft_plan *plan;

	plan = new curafft_plan();
    memset(plan, 0, sizeof(curafft_plan));

	int direction = 1; //inverse
	
	// opts and copts setting
    plan->opts.gpu_device_id = 0;
    plan->opts.upsampfac = sigma;
    plan->opts.gpu_sort = 1;
    plan->opts.gpu_binsizex = -1;
    plan->opts.gpu_binsizey = -1;
    plan->opts.gpu_binsizez = -1;
    plan->opts.gpu_kerevalmeth = kerevalmeth;
    plan->opts.gpu_conv_only = 0;
    plan->opts.gpu_gridder_method = method;

    ier = setup_conv_opts(plan->copts, epsilon, sigma, 1, direction, kerevalmeth); //check the arguements
	
	if(ier!=0)printf("setup_error\n");

    // plan setting
    // cuda stream malloc in setup_plan
    

    int nf1 = get_num_cells(N*2,plan->copts);
	//printf("nf: %d\n",nf1);
    //printf("copt info kw %d, upsampfac %lf, beta %lf\n",plan->copts.kw,plan->copts.upsampfac,plan->copts.ES_beta);
    plan->dim = 1;
    setup_plan(nf1, 1, 1, M, d_u, NULL, NULL, d_c, plan);

	plan->ms = N; ///!!!
	plan->mt = 1;
	plan->mu = 1;
    plan->execute_flow = 1;
	int iflag = direction;
    int fftsign = (iflag>=0) ? 1 : -1;

	plan->iflag = fftsign; //may be useless| conflict with direction
	plan->batchsize = 1;

    plan->copts.direction = direction; // 1 inverse, 0 forward
    PCS *d_fwkerhalf;
    checkCudaErrors(cudaMalloc((void**)&d_fwkerhalf,sizeof(PCS)*(N)));
	cudaMemset(d_fwkerhalf,0,sizeof(PCS)*N);
    PCS *d_k;
    checkCudaErrors(cudaMalloc((void**)&d_k,sizeof(PCS)*(N)));
    checkCudaErrors(cudaMemcpy(d_k,k,sizeof(PCS)*(N),cudaMemcpyHostToDevice));
    fourier_series_appro_invoker(d_fwkerhalf,d_k,plan->copts, N,nf1/2+1); // correction with k, may be wrong, k will be free in this function
	
	
	// printf("begining...\n");
	// fourier_series_appro_invoker(d_fwkerhalf,plan->copts,nf1/2+1);
	PCS *fwkerhalf = (PCS *)malloc(sizeof(PCS)*(N));
	cudaMemcpy(fwkerhalf, d_fwkerhalf, sizeof(PCS)*(N), cudaMemcpyDeviceToHost);
#ifdef DEBUG
	printf("correction factor printing method1...\n");
	for (int i = 0; i < N; i++)
	{
		/* code */
		printf("%lf ",fwkerhalf[i]);
	}
	printf("\n");
#endif
	// fw (conv res set)
	checkCudaErrors(cudaMalloc((void**)&d_fw,sizeof(CUCPX)*nf1));
	checkCudaErrors(cudaMemset(d_fw, 0, sizeof(CUCPX)*nf1));
	plan->fw = d_fw;
	// fk malloc and set
	checkCudaErrors(cudaMalloc((void**)&d_fk,sizeof(CUCPX)*N));
	plan->fk = d_fk;

	// calulating result
	curafft_conv(plan);
	CPX *fw = (CPX *)malloc(sizeof(CPX)*nf1);
	cudaMemcpy(fw,plan->fw,sizeof(CUCPX)*nf1,cudaMemcpyDeviceToHost);
#ifdef DEBUG
	printf("conv result printing...\n");
	
	for (int i = 0; i < nf1; i++)
	{
		/* code */
		printf("%lf ",fw[i].real());
	}
	printf("\n");
	
#endif
	CPX *fk = (CPX *)malloc(sizeof(CPX)*N);
	memset(fk,0,sizeof(CPX)*N);
	// dft
	for (int i = 0; i < N; i++)
	{
		/* code */
		for (int j = 0; j < nf1; j++)
		{
			if(j<nf1/2){
                fk[i] += fw[j+nf1/2]*exp(k[i]*((j)/((PCS)nf1)*2.0*PI*IMA));
            }
            else{
                fk[i] += fw[j-nf1/2]*exp(k[i]*( (j-(PCS)nf1)/((PCS)nf1) )*2.0*PI*IMA); //fw[j-nf1/2]*exp(k[i]*( (j-nf1)/((PCS)nf1) )*2.0*PI*IMA); not work why
            }
		}
		
	}
#ifdef DEBUG
	printf("dft result printing...\n");
	for (int i = 0; i < N; i++)
	{
		/* code */
		printf("%lf ",fk[i].real());
	}
	printf("\n");
#endif
	

	// printf("correction factor printing...\n");
	// for(int i=0; i<N1/2; i++){
	// 	printf("%.3g ",fwkerhalf1[i]);
	// }
	// printf("\n");
	// for(int i=0; i<N2/2; i++){
	// 	printf("%.3g ",fwkerhalf2[i]);
	// }
	// printf("\n");
	// deconv
	//PCS *fwkerhalf = (PCS *)malloc(sizeof(PCS)*(N));
	//cudaMemcpy(fwkerhalf, d_fwkerhalf, sizeof(PCS)*(N), cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++){
		fk[i] = fk[i] / fwkerhalf[i];
	}

	
	// result printing
	printf("final result printing...\n");
	for(int i=0; i<10; i++){
		printf("%.10lf ",fk[i].real());
		
	}
	printf("\n");
	CPX *truth = (CPX *) malloc(sizeof(CPX)*N);
	printf("ground truth printing...\n");
	for (int i = 0; i < N; i++)
	{
		truth[i] = 0;
		for (int j = 0; j < M; j++)
		{
			truth[i] += c[j] * exp(k[i] * u[j] * IMA);
		}
	}

	for (int i = 0; i < 10; i++)
	{
		printf("%.10lf ", truth[i].real());
	}
	printf("\n");


	double max=0;
	double l2_max=0;
	//double fk_max = 0;
	// for(int i=0; i<M; i++){
	// 	if(abs(fk[i].real())>fk_max)fk_max = abs(fk[i].real());
	// }
	// printf("fk max %lf\n",fk_max);
	for(int i=0; i<N; i++){
		double temp = abs(truth[i].real()-fk[i].real());
		if(temp>max) max = temp;
		if(temp/fk[i].real() > l2_max) l2_max = temp/fk[i].real();
	}
	printf("maximal abs error %.6g, maximal l2 error %.6g\n",max,l2_max);
	
	//free
	curafft_free(plan);
	free(fk);
	free(u);
	free(c);

	return ier;
}