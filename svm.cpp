/* @file: svm.cpp
 * @brief: a prototype program of support vector machine, for learning
 * @reference: Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin, 2005, Working Set Selection Using Second Order Information for Training Support Vector Machines, Journal of Machine Learning Research, 1889-1918.
 * @addtional: use WSS3 algorithm as the working set selection algortihm, implement an SMO type algorithm.
 */
#include <iostream>
//#include <mkl.h>
#include <mm_malloc.h>
#include <omp.h>
#include <string.h>
#include <assert.h>

#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#include <mmintrin.h> // MMX
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1
#include <nmmintrin.h> // SSE4.2
#include <immintrin.h> // AVX
//#include <intrin.h> 

#include <time.h>

using namespace std;

const int Aligned = 16;
#define AVX

#ifdef DEBUG
double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
#endif

template<typename T>
T * ArrayAlloc1D(const int n0) {
	T * ptr = (T*)_mm_malloc(sizeof(T) * n0, Aligned);
	memset((void*)ptr, 0, sizeof(T) * n0);
	return ptr;
}

template<typename T>
void ArrayFree1D(T * ptr) {
	_mm_free(ptr);
}

template<typename T>
void ArrayAlloc2D(T*** const ptr, const int n0, const int n1) {
	*ptr = (T**)malloc(sizeof(T*) * n1);
	(*ptr)[0] = (T*)_mm_malloc(sizeof(T) * n0 * n1, Aligned);
	for(int i = 1; i < n1; i++) {
		(*ptr)[i] = (*ptr)[0] + i * n0;
	}
}

template<typename T>
void ArrayFree2D(T** ptr) {
	_mm_free(ptr[0]);
	free(ptr);
}

template<typename T>
void ArrayAlloc3D(T**** const ptr, const int n0, const int n1, const int n2) {
	*ptr = (T***)malloc(sizeof(T**) * n2);
	(*ptr)[0] = (T**)malloc(sizeof(T*) * n1 * n2);
	for(int j = 1; j < n2; j++) (*ptr)[j] = (*ptr)[0] + j * n1;
	(*ptr)[0][0] = (T*)_mm_malloc(sizeof(T) * n0 * n1 * n2, Aligned);
	for(int i = 1; i < n1; i++) (*ptr)[0][i] = (*ptr)[0][0] + i * n0;
	for(int i = 1; i < n2; i++) {
		for(int j = 0; j < n1; j++) {
			(*ptr)[i][j] = (*ptr)[0][0] + i * n1 * n0 + j * n0;
		}
	}	
}

template<typename T>
void ArrayFree3D(T*** ptr) {
	_mm_free(ptr[0][0]);
	free(ptr[0]);
	free(ptr);
}

class KernelFunction {
public:
	KernelFunction() { };
	~KernelFunction() { };
	inline virtual float compute(const float * Xi, const float * Xj, const int ndims) const = 0;
//	inline virtual float compute(const float * Xi, const float * Xj, const int ndims) const { return 1.0; };
};

class LinearKernel : public KernelFunction {
public:
	LinearKernel() { };
	~LinearKernel() { };
	inline virtual float compute(const float * Xi, const float * Xj, const int ndims) const;
};

inline float LinearKernel::compute(const float * Xi, const float * Xj, const int ndims) const
{
	float ret = 0.0;
	int off = 0;
	#ifdef AVX
	for(; off <= ndims - 8; off += 8, Xi += 8, Xj += 8) {
		__m256 __xi = _mm256_loadu_ps(Xj);
		__m256 __xj = _mm256_loadu_ps(Xi);
		__m256 __ret = _mm256_mul_ps(__xi, __xj);
		__ret = _mm256_hadd_ps(__ret, __ret);
		__ret = _mm256_hadd_ps(__ret, __ret);
	//	float temp[8] __attribute__((aligned(Aligned)));
	//	_mm256_store_ps(temp, __ret);
	//	ret += temp[0];
	//	ret += temp[1];
	//	ret += temp[2];
	//	ret += temp[3];
	//	ret += temp[4];
	//	ret += temp[5];
	//	ret += temp[6];
	//	ret += temp[7];
	
		__m128 __f = _mm_load_ss(&ret); __f = _mm_shuffle_ps(__f, __f, 0);
		__f = _mm_add_ps(__f, _mm_add_ps(_mm256_extractf128_ps(__ret, 0), _mm256_extractf128_ps(__ret, 1)));
		_mm_store_ss(&ret, __f);
		
	}
	#endif
	for(; off < ndims; off++, Xi++, Xj++) {
		ret += (*Xi) * (*Xj);
	}
	return ret;
}

class RBFKernel : public KernelFunction {
public:
	RBFKernel(float gamma_) : gamma(gamma_) { };
	~RBFKernel() { };
	inline virtual float compute(const float * Xi, const float * Xj, const int ndims) const;
public:
	float gamma;
};

inline float RBFKernel::compute(const float * Xi, const float * Xj, const int ndims) const
{
	float ret = 0.0;
	int off = 0;
	#ifdef AVX
	for(; off <= ndims - 8; off += 8, Xi += 8, Xj += 8) {
		__m256 __xi = _mm256_loadu_ps(Xj);
		__m256 __xj = _mm256_loadu_ps(Xi);
		__m256 __ret = _mm256_sub_ps(__xi, __xj);
		__ret = _mm256_mul_ps(__ret, __ret);
		__ret = _mm256_hadd_ps(__ret, __ret);
		__ret = _mm256_hadd_ps(__ret, __ret);
	//	float temp[8] __attribute__((aligned(Aligned)));
	//	_mm256_store_ps(temp, __ret);
	//	float t = temp[0] + temp[3];
	//	ret += t;
	
		__m128 __f = _mm_load_ss(&ret); __f = _mm_shuffle_ps(__f, __f, 0);
		__f = _mm_add_ps(__f, _mm_add_ps(_mm256_extractf128_ps(__ret, 0), _mm256_extractf128_ps(__ret, 1)));
		_mm_store_ss(&ret, __f);
		
		
	}
	#endif
	for(; off < ndims; off++, Xi++, Xj++) {
		ret += (*Xi - *Xj) * (*Xi - *Xj);
	}
	return expf(- gamma * ret);
}

class PolynomialKernel : public KernelFunction {
public:
	PolynomialKernel(float gamma_, float d_) : gamma(gamma_), d(d_) { };
	~PolynomialKernel() { };
	inline virtual float compute(const float * Xi, const float * Xj, const int ndims) const;
public:
	float gamma;
	float d;
};

inline float PolynomialKernel::compute(const float * Xi, const float * Xj, const int ndims) const
{
	float ret = 1.0;
	int off = 0;
	#ifdef AVX
	for(; off <= ndims - 8; off += 8, Xi += 8, Xj += 8) {
		__m256 __xi = _mm256_loadu_ps(Xj);
		__m256 __xj = _mm256_loadu_ps(Xi);
		__m256 __ret = _mm256_mul_ps(__xi, __xj);
		__ret = _mm256_hadd_ps(__ret, __ret);
		__ret = _mm256_hadd_ps(__ret, __ret);
	//	float temp[8] __attribute__((aligned(Aligned)));
	//	_mm256_store_ps(temp, __ret);
	//	float t = temp[0] + temp[3];
	//	ret += t;
	
		__m128 __f = _mm_load_ss(&ret); __f = _mm_shuffle_ps(__f, __f, 0);
		__f = _mm_add_ps(__f, _mm_add_ps(_mm256_extractf128_ps(__ret, 0), _mm256_extractf128_ps(__ret, 1)));
		_mm_store_ss(&ret, __f);
		
		
	}
	#endif
	for(; off < ndims; off++, Xi++, Xj++) {
		ret += (*Xi) * (*Xj);
	}
	return powf((gamma * ret), d);
}

class SigmoidKernel : public KernelFunction {
public:
	SigmoidKernel(float gamma_, float d_) : gamma(2.f * gamma_), d(2.f * d_) { };
	~SigmoidKernel() { };
	inline virtual float compute(const float * Xi, const float * Xj, const int ndims) const;
public:
	float gamma;
	float d;
};

inline float SigmoidKernel::compute(const float * Xi, const float * Xj, const int ndims) const
{
	float ret = 0.0;
	int off = 0;
	#ifdef AVX
	for(; off <= ndims - 8; off += 8, Xi += 8, Xj += 8) {
		__m256 __xi = _mm256_loadu_ps(Xj);
		__m256 __xj = _mm256_loadu_ps(Xi);
		__m256 __ret = _mm256_mul_ps(__xi, __xj);
		__ret = _mm256_hadd_ps(__ret, __ret);
		__ret = _mm256_hadd_ps(__ret, __ret);
	//	float temp[8] __attribute__((aligned(Aligned)));
	//	_mm256_store_ps(temp, __ret);
	//	float t = temp[0] + temp[3];
	//	ret += t;
	
		__m128 __f = _mm_load_ss(&ret); __f = _mm_shuffle_ps(__f, __f, 0);
		__f = _mm_add_ps(__f, _mm_add_ps(_mm256_extractf128_ps(__ret, 0), _mm256_extractf128_ps(__ret, 1)));
		_mm_store_ss(&ret, __f);
		
		
	}
	#endif
	for(; off < ndims; off++, Xi++, Xj++) {
		ret += (*Xi) * (*Xj);
	}
	ret = expf(gamma * ret + d);
	return (ret - 1.0) / (ret + 1.0);
}

class SupportVectorMachine {
public:
	SupportVectorMachine() : X(nullptr), y(nullptr), array(nullptr), b(0.0), nSamples(0), nDims(0), C(0.0), eps(1e-3) { };
	SupportVectorMachine(float ** X_, int * y_, const int nSamples_, const int nDims_, const float C_, const float eps_ = 1e-3) : X(X_), y(y_), nSamples(nSamples_), nDims(nDims_) , C(C_), eps(eps_) { 
		assert(nSamples > 0 && nDims > 0 && C > 0);
		assert(X != nullptr && y != nullptr);
		array = ArrayAlloc1D<float>(nSamples);
		b = 0.0;		
	}
	~SupportVectorMachine() {
		X = nullptr;
		y = nullptr;
		if(array != nullptr) {
			ArrayFree1D<float>(array);
			array = nullptr;
		}
	}
	float getC() { return C; };
	void setC(const float C_) { C = C_; };
	float getEps() { return eps; };
	void setEps(const float eps_) { eps = eps_; };
	void setTrainData(float ** X_, int * y_) { X = X_; y = y_; };
	
	int getDims() { return nDims; };
	void setDims(const int nDims_) { nDims = nDims_; };
	int getSamples() { return nSamples; };
	void setSamples(const int nSamples_) { 
		if(nSamples != nSamples_) {
			if(array != nullptr) {
				ArrayFree1D<float>(array); 
				array = nullptr;
			}
			nSamples = nSamples_; 
			array = ArrayAlloc1D<float>(nSamples); 
		}
	}

	void train();
	void train(KernelFunction * func);
	void predict(int * yt, float * const * Xt, const int nobjs);
	void predict(int * yt, float * const * Xt, const int nobjs, KernelFunction * func);
	void crossValidation(const int nFolds);
	void clearModel() {
		X = nullptr;
		y = nullptr;
		if(array != nullptr) {
			ArrayFree1D<float>(array);
			array = nullptr;
		}
		b = 0.0;
		nSamples = 0;
		nDims = 0;
		C = 0.0;
		eps = 1e-3;	
	}
private:
	void randPerm();
	void computeKernelMatrix(float * Q);
	void computeKernelMatrixRow(float * Q, const int i, KernelFunction * func);
	void computeKernelMatrixDiag(float * Q_diag, KernelFunction * func);
	void updateGradient(float * G, const float * Q, const float dWi, const float dWj, const int i, const int j);
	void updateGradient(float * G, float * const * Q, const float dWi, const float dWj);
	void selectWorkingSet(int &i, int &j, const float * G, const float * Q);
	void selectWorkingSet(int &i, int &j, const float * G, const float * Q_diag, float ** Qij, KernelFunction * func);
protected:
	float ** X;
	int * y;
	float * array;
	float b;
	
	int nSamples;
	int nDims;
	float C;
	float eps;
public:
	static const float tau;
};

const float SupportVectorMachine::tau = 1e-12;

#define cut(w, c) ((w) <= (c) ? ((w) >= 0 ? (w) : 0) : (c))
void SupportVectorMachine::train() 
{
	assert(nSamples > 0 && nDims > 0);
	assert(X != nullptr && y != nullptr && array != nullptr);

	float * Q = (float*)_mm_malloc(sizeof(float) * nSamples * nSamples, Aligned);
	float * G = (float*)_mm_malloc(sizeof(float) * nSamples, Aligned);
	memset(array, 0, sizeof(float) * nSamples);

	computeKernelMatrix(Q);	

	for(int i = 0; i < nSamples; i++) G[i] = -1.0;

	#ifdef DEBUG
	int iteration = 0;
	#endif

	while(1) {
		#ifdef DEBUG
	//	fprintf(stderr, "\nDEBUG: iteration step %d\n", iteration);
		#endif
		int i, j;
		selectWorkingSet(i, j, G, Q);
		#ifdef DEBUG
	//	fprintf(stderr, "DEBUG: iteration step %d: working set (%d, %d)\n", iteration, i, j);
	//	fprintf(stderr, "\n");
		iteration++;
		#endif	
		if(j == -1) break;

		// working set is (i, j)
		float qii = Q[i * nSamples + i];
		float qjj = Q[j * nSamples + j];
		float qij = Q[i * nSamples + j];
		float a = qii + qjj - 2.0 * y[i] * y[j] * qij;
		if(a <= 0) a = SupportVectorMachine::tau;
		float b = - y[i] * G[i] + y[j] * G[j];		
		
		// update weight	
		float oldAi = array[i];
		float oldAj = array[j];
		array[i] += y[i] * b / a;
		array[j] -= y[j] * b / a;
			
		// project alpha back to the feasible region
		float sum = y[i] * oldAi + y[j] * oldAj;
		array[i] = cut(array[i], C);
		array[j] = y[j] * (sum - y[i] * array[i]);
		array[j] = cut(array[j], C);
		array[i] = y[i] * (sum - y[j] * array[j]);
	
		// update gradient
		float dAi = array[i] - oldAi;
		float dAj = array[j] - oldAj;
		updateGradient(G, Q, dAi, dAj, i, j);			
	}

	const float tol = 1e-6;
	for(int idx = 0; idx < nSamples; idx++) {
		if(array[idx] > tol || array[idx] < -tol) {
			float yi = y[idx];
			float wx = 0.0;
			for(int jj = 0; jj < nSamples; jj++) {
				wx += array[jj] * Q[idx * nSamples + jj] / yi;
			}
			b = yi - wx;	
		}
	}

	for(int idx = 0; idx < nSamples - 1; idx++) {
		cout << array[idx] << "\t";
	}
	cout << array[nSamples - 1] << "\n";

	// cout << array[0] << "\t" << array[1] << "\t" << array[2] << "..." << array[nSamples - 1] << "\n";
	#ifdef DEBUG
	cout << "iteration = " << iteration << "\n";
	#endif

	_mm_free(Q);
	_mm_free(G);	
}

void SupportVectorMachine::computeKernelMatrix(float * Q)
{
	for(int i = 0; i < nSamples; i++) {
		float yi = y[i];
		const int * yj_ptr = y;
		const float * Xi_ptr = X[i];
		const float * Xj_ptr = X[0];
		float * Q_ptr = Q + i * nSamples;
		for(int j = 0; j < nSamples; j++, yj_ptr++, Xj_ptr += nDims, Q_ptr++) {
			float yj = *yj_ptr;
			float kij = 0.0;
			for(int k = 0; k < nDims; k++) {
				kij += Xi_ptr[k] * Xj_ptr[k];
			}
			*Q_ptr = yi * yj * kij;		
		}
	}
}

void SupportVectorMachine::updateGradient(float * G, const float * Q, const float dAi, const float dAj, const int i, const int j)
{
	const float * Qi_ptr = &Q[i * nSamples];
	const float * Qj_ptr = &Q[j * nSamples];
	for(int k = 0; k < nSamples; k++, Qi_ptr++, Qj_ptr++) {
		G[k] += (*Qi_ptr) * dAi + (*Qj_ptr) * dAj;
	}	
}

const float Infinity = 1e14;
void SupportVectorMachine::selectWorkingSet(int &i, int &j, const float * G, const float * Q)
{
	const float Gap = 0.0;

	i = -1;
	float G_max = -Infinity;
	float G_min = Infinity;

	// select i
	for(int k = 0; k < nSamples; k++) {
		if((y[k] == 1 && array[k] < C - Gap) || (y[k] == -1 && array[k] > 0 + Gap)) {
			float yg = - y[k] * G[k];
			if(yg >= G_max) {
				i = k;
				G_max = yg;
			}
		} 
	}
	
	// select j
	j = -1;
	// TODO: careful
	float qii = Q[0];
	const float * Qi_ptr = Q;
	float yi = y[0];
	if(i >= 0) {
		qii = Q[i * nSamples + i];
		Qi_ptr += i * nSamples;
		yi = y[i];
	}
	float obj_min = Infinity;
	for(int k = 0; k < nSamples; k++, Qi_ptr++) {
		if((y[k] == 1 && array[k] > 0 + Gap) || (y[k] == -1 && array[k] < C - Gap)) {
			float yg = y[k] * G[k];
			float b = G_max + yg;
			G_min = -yg <= G_min ? -yg : G_min;
			if(b > 0) {
				float a = qii + Q[k * nSamples + k] - 2.0 * yi * y[k] * (*Qi_ptr);
				a = a > 0 ? a : SupportVectorMachine::tau;
				float bba = - b * b / a;
				if(bba <= obj_min) {
					j = k;
					obj_min = bba;
				}
			}	
				
		}
	}
	#ifdef DEBUG
//	fprintf(stderr, "DEBUG: G_max = %f, G_min = %f\n", G_max, G_min);
	#endif
	if(G_max - G_min < eps) {
		i = -1; j = -1;
		return;
	}	
	return;
}

void SupportVectorMachine::computeKernelMatrixRow(float * Q, const int i, KernelFunction * func)
{
//	const int threadNum = omp_get_max_threads();
//	const int threadNum = 4;
	const float * Xi_ptr = X[i];
	float yi = y[i];
//	#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(Xi_ptr, X)
	for(int k = 0; k < nSamples; k++) {
		Q[k] = func->compute(Xi_ptr, X[k], nDims);
		Q[k] *= yi * y[k];
	}
}

void SupportVectorMachine::computeKernelMatrixDiag(float * Q_diag, KernelFunction * func)
{
//	const int threadNum = omp_get_max_threads();
//	const int threadNum = 4;
//	#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(X)
	for(int k = 0; k < nSamples; k++) {
		Q_diag[k] = func->compute(X[k], X[k], nDims);
		Q_diag[k] *= y[k] * y[k];
	}
} 

void SupportVectorMachine::updateGradient(float * G, float * const * Qij, const float dAi, const float dAj)
{
	const float * Qi_ptr = Qij[0];
	const float * Qj_ptr = Qij[1];
	int k = 0;
	#ifdef AVX
	__m256 dai = _mm256_broadcast_ss(&dAi);
	__m256 daj = _mm256_broadcast_ss(&dAj);
	for(; k <= nSamples - 8; k += 8) {
		__m256 g = _mm256_loadu_ps(&G[k]);
		__m256 qi = _mm256_loadu_ps(&Qi_ptr[k]);
		__m256 qj = _mm256_loadu_ps(&Qj_ptr[k]);
		qi = _mm256_mul_ps(qi, dai);
		qj = _mm256_mul_ps(qj, daj);
		g = _mm256_add_ps(g, _mm256_add_ps(qi, qj));
		_mm256_storeu_ps(&G[k], g);		
	}
	#endif
	for(; k < nSamples; k++) {
		G[k] += Qi_ptr[k] * dAi + Qj_ptr[k] * dAj;	
	}
}

void SupportVectorMachine::selectWorkingSet(int &i, int &j, const float * G, const float * Q_diag, float ** Qij, KernelFunction * func)
{
	float * Qi = Qij[0];
	float * Qj = Qij[1];
	memset(Qi, 0, sizeof(float) * nSamples * 2);	

	const float Gap = 0.0;

	i = -1;
	float G_max = -Infinity;
	float G_min = Infinity;

	// select i
	for(int k = 0; k < nSamples; k++) {
		if((y[k] == 1 && array[k] < C - Gap) || (y[k] == -1 && array[k] > 0 + Gap)) {
			float yg = - y[k] * G[k];
			if(yg >= G_max) {
				i = k;
				G_max = yg;
			}
		} 
	}

	#ifdef debug
	t0 = omp_get_wtime();
	#endif
	if(i >= 0) {
		computeKernelMatrixRow(Qi, i, func);	
	}
	#ifdef DEBUG
	t4 += omp_get_wtime() - t0;
	#endif
	
	// select j
	j = -1;
	// TODO: careful
	float qii = Q_diag[0];
	float yi = y[0];
	if(i >= 0) {
		qii = Q_diag[i];
		yi = y[i];
	}
	float obj_min = Infinity;
	for(int k = 0; k < nSamples; k++) {
		if((y[k] == 1 && array[k] > 0 + Gap) || (y[k] == -1 && array[k] < C - Gap)) {
			float yg = y[k] * G[k];
			float b = G_max + yg;
			G_min = -yg <= G_min ? -yg : G_min;
			if(b > 0) {
				float a = qii + Q_diag[k] - 2.0 * yi * y[k] * Qi[k];
				a = a > 0 ? a : SupportVectorMachine::tau;
				float bba = - b * b / a;
				if(bba <= obj_min) {
					j = k;
					obj_min = bba;
				}
			}	
				
		}
	}
	#ifdef DEBUG
	fprintf(stderr, "DEBUG: G_max = %f, G_min = %f\n", G_max, G_min);
	#endif
	if(G_max - G_min < eps) {
		i = -1; j = -1;
		return;
	}

	#ifdef debug
	t0 = omp_get_wtime();
	#endif
	if(j >= 0) {
		computeKernelMatrixRow(Qj, j, func);
	} 	
	#ifdef debug
	t4 += omp_get_wtime() - t0;
	#endif
	return;
}

void SupportVectorMachine::train(KernelFunction * func) 
{
	assert(nSamples > 0 && nDims > 0);
	assert(X != nullptr && y != nullptr && array != nullptr);

	float * Q_diag = (float*)_mm_malloc(sizeof(float) * nSamples, Aligned);
	float * G = (float*)_mm_malloc(sizeof(float) * nSamples, Aligned);
	
	float ** Qij;
	ArrayAlloc2D<float>(&Qij, nSamples, 2);
	memset(array, 0, sizeof(float) * nSamples);

	#ifdef DEBUG
	t0 = omp_get_wtime();
	#endif
	computeKernelMatrixDiag(Q_diag, func); 
	#ifdef DEBUG
	t1 += omp_get_wtime() - t0;
	#endif
	
	for(int i = 0; i < nSamples; i++) G[i] = -1.0;

	#ifdef DEBUG
	int iteration = 0;
	#endif

	while(1) {
		#ifdef DEBUG
		fprintf(stderr, "\nDEBUG: iteration step %d\n", iteration);
		#endif
		int i, j;
		#ifdef DEBUG
		t0 = omp_get_wtime();
		#endif
		selectWorkingSet(i, j, G, Q_diag, Qij, func);
		#ifdef DEBUG
		t2 += omp_get_wtime() - t0;
		#endif
		#ifdef DEBUG
		fprintf(stderr, "DEBUG: iteration step %d: working set (%d, %d)\n", iteration, i, j);
		fprintf(stderr, "\n");
		iteration++;
		#endif	
		if(j == -1) break;

		// working set is (i, j)
		float qii = Q_diag[i];
		float qjj = Q_diag[j];
		float qij = Qij[0][j];
		float a = qii + qjj - 2.0 * y[i] * y[j] * qij;
		if(a <= 0) a = SupportVectorMachine::tau;
		float b = - y[i] * G[i] + y[j] * G[j];		
		
		// update weight	
		float oldAi = array[i];
		float oldAj = array[j];
		array[i] += y[i] * b / a;
		array[j] -= y[j] * b / a;
			
		// project alpha back to the feasible region
		float sum = y[i] * oldAi + y[j] * oldAj;
		array[i] = cut(array[i], C);
		array[j] = y[j] * (sum - y[i] * array[i]);
		array[j] = cut(array[j], C);
		array[i] = y[i] * (sum - y[j] * array[j]);
	
		// update gradient
		float dAi = array[i] - oldAi;
		float dAj = array[j] - oldAj;
		
		#ifdef DEBUG
		t0 = omp_get_wtime();
		#endif
		updateGradient(G, Qij, dAi, dAj);
		#ifdef DEBUG
		t3 += omp_get_wtime() - t0;
		#endif
	}

	const float tol = 1e-14;
	for(int idx = 0; idx < nSamples; idx++) {
		if(array[idx] > tol || array[idx] < -tol) {
			float yi = y[idx];
			float wx = 0.0;
			for(int jj = 0; jj < nSamples; jj++) {
				float kij = func->compute(X[jj], X[idx], nDims);
				wx += array[jj] * y[jj] * kij;
			}
			b = yi - wx;	
		}
	}

	for(int idx = 0; idx < nSamples - 1; idx++) {
		cout << array[idx] << "\t";
	}
	cout << array[nSamples - 1] << "\n";
//	cout << array[0] << "\t" << array[1] << "\t" << array[2] << "..." << array[nSamples - 1] << "\n";

	#ifdef DEBUG
	cout << "t1 = " << t1 << "s \n";
	cout << "t2 = " << t2 << "s \n";
	cout << "t3 = " << t3 << "s \n";
	cout << "t4 = " << t4 << "s \n";
	cout << "iteration = " << iteration << "\n";
	#endif	

	ArrayFree2D<float>(Qij);
	Qij = nullptr;
	_mm_free(Q_diag);
	_mm_free(G);	
}

void SupportVectorMachine::predict(int * yt, float * const * Xt, const int nobjs)
{
	LinearKernel lk;
	predict(yt, Xt, nobjs, &lk);
}

void SupportVectorMachine::predict(int * yt, float * const * Xt, const int nobjs, KernelFunction * func)
{
	for(int i = 0; i < nobjs; i++) {
		float wx = 0.0;
		for(int j = 0; j < nSamples; j++) {
			float kij = func->compute(Xt[i], X[j], nDims);
			wx += array[j] * y[j] * kij;	
		}
		wx += b;
		yt[i] = wx > 0 ? 1 : -1;
	}	
}

/* simple class to read csv file 
 * assume all attributes are read number varibles, and the label is object varible
 */
class CSVFile {
public:
	CSVFile() : X(nullptr) { };
	CSVFile(const string& filename) : m_fil(filename) {
		m_fhl.open(filename, ios::in);
		if(!m_fhl.good()) {
			fprintf(stderr, "ERROR: cannot open csv file %s\n", m_fil.c_str());
			exit(-1);
		}
		string line;
		getline(m_fhl, line);
		Split(line, ",", m_features);
		m_ndims = m_features.size();
		m_nsamples = 0;
		while(getline(m_fhl, line)) {
			m_nsamples++;
		}
		// rewind	
		m_fhl.seekg(0);	
	}
	static void Split(const string& str, const string& sep, vector<string>& ret);
	virtual ~CSVFile() {
		m_fhl.close();
		ArrayFree2D<float>(X);
		X = nullptr;
	}
	virtual void read() {
		assert(m_nsamples > 0 && m_ndims > 0);
		ArrayAlloc2D<float>(&X, m_ndims, m_nsamples);
		assert(m_fhl.good());
		m_fhl.seekg(0);
		string line;
		getline(m_fhl, line);
		float * X_ptr = X[0];
		for(int iline = 0; iline < m_nsamples; iline++, X_ptr += m_ndims) {
			getline(m_fhl, line);
			vector<string> splitLine;
			Split(line, ",", splitLine);
			assert(splitLine.size() == (unsigned)m_ndims);
			for(int ifeat = 0; ifeat < m_ndims; ifeat++) {
				sscanf(splitLine[ifeat].c_str(), "%f", &X_ptr[ifeat]);
			}
		}
	}
	virtual void printData() {
		for(int i = 0; i < m_ndims - 1; i++) {
			printf("%s\t", m_features[i].c_str());
		}
		printf("%s\n", m_features[m_ndims - 1].c_str());
		for(int i = 0; i < m_nsamples; i++) {
			for(int j = 0; j < m_ndims - 1; j++) {
				printf("%f\t", X[i][j]);
			}
			printf("%f\n", X[i][m_ndims - 1]);
		}
	}
public:
	string m_fil;
	ifstream m_fhl;
	vector<string> m_features;
	float ** X;
	int m_nsamples;
	int m_ndims;
};

class CSVTrain : public CSVFile {
public:
	CSVTrain() : y(nullptr) { };
	CSVTrain(const string& filename, const string& target) : m_target(target) {
		m_fil = filename;
		m_fhl.open(filename, ios::in);
		if(!m_fhl.good()) {
			fprintf(stderr, "ERROR: cannot open csv file %s\n", m_fil.c_str());
			exit(-1);
		}
		string line;
		getline(m_fhl, line);
		Split(line, ",", m_features);
		auto ite = find(m_features.begin(), m_features.end(), m_target);
		if(ite == m_features.end()) {
			fprintf(stderr, "ERROR: target<%s> is not in the first line of the csv file %s\n", m_target.c_str(), m_fil.c_str());
			exit(-2);
		}
		targetIdx = ite - m_features.begin();
		m_features.erase(ite);
		m_ndims = m_features.size();
		m_nsamples = 0;
		while(getline(m_fhl, line)) {
			m_nsamples++;
		}
		m_fhl.clear();
		// rewind	
		m_fhl.seekg(0);
	}
	virtual ~CSVTrain() {
		ArrayFree1D<int>(y);
		y = nullptr;
	}
	virtual void read() {
		assert(m_nsamples > 0 && m_ndims > 0);
		ArrayAlloc2D<float>(&X, m_ndims, m_nsamples);
		y = ArrayAlloc1D<int>(m_nsamples);
		assert(m_fhl.good());
		m_fhl.seekg(0);
		string line;
		getline(m_fhl, line);
		float * X_ptr = X[0];
		int nObjs = 0;
		for(int iline = 0; iline < m_nsamples; iline++, X_ptr += m_ndims) {
			getline(m_fhl, line);
			vector<string> splitLine;
			Split(line, ",", splitLine);
			assert(splitLine.size() == (unsigned)(m_ndims + 1));
			for(int ifeat = 0; ifeat < targetIdx; ifeat++) {
				sscanf(splitLine[ifeat].c_str(), "%f", &X_ptr[ifeat]);
			}
			auto search = objMap.find(splitLine[targetIdx]);
			if(search == objMap.end()) {
				objMap.insert(make_pair(splitLine[targetIdx], nObjs));
				labels.push_back(splitLine[targetIdx]);
				y[iline] = nObjs;
				nObjs++;
			} else {
				y[iline] = search->second;
			}
			for(int ifeat = targetIdx + 1; ifeat < m_ndims + 1; ifeat++) {
				sscanf(splitLine[ifeat].c_str(), "%f", &X_ptr[ifeat - 1]);
			}
		}
		
	}
	virtual void printData() {
		for(int i = 0; i < m_ndims; i++) {
			printf("%s\t", m_features[i].c_str());
		}
		printf("%s\n", m_target.c_str());
		for(int i = 0; i < m_nsamples; i++) {
			for(int j = 0; j < m_ndims; j++) {
				printf("%f\t", X[i][j]);
			}
			printf("%d\n", y[i]);
//			printf("%s\n", labels[y[i]].c_str());
		}
	}
public:
	string m_target;
	int targetIdx;
	vector<string> labels;
	unordered_map<string, int> objMap;
	int * y;
};

class CSVTest : public CSVFile {
public:
};

void CSVFile::Split(const string& str, const string& sep, vector<string>& ret)
{
	size_t len = str.size();
	size_t first = 0;
	size_t index = str.find_first_of(sep, first);
	while(index != string::npos) {
		ret.push_back(str.substr(first, index - first));
		first = index + 1;
		if(first >= len) break;
		index = str.find_first_of(sep, first);
	}
	if(first < len) {
		ret.push_back(str.substr(first));
	}
}

int main(int argc, char * argv[])
{
	if(argc < 2) {
		fprintf(stdout, "Usage: inputcsvfile\n");
		return -1;
	}
	CSVTrain train(argv[1], "Diagnosis");
	train.read();
//	train.printData();
	SupportVectorMachine svm(train.X, train.y, train.m_nsamples, train.m_ndims, 10.0, 1e-5);
	for(int i = 0; i < train.m_nsamples; i++) train.y[i] = train.y[i] == 0 ? -1 : train.y[i];
	LinearKernel lk;
	RBFKernel rbfk(100);
	PolynomialKernel pk(1.0, 2.0);
	double t0;
	t0 = omp_get_wtime();
	svm.train(&rbfk);
	t0 = omp_get_wtime() - t0;
	fprintf(stdout, "elapsed time of svm(AVX) is %f s\n", t0);

	int * yt = ArrayAlloc1D<int>(train.m_nsamples);
	svm.predict(yt, train.X, train.m_nsamples, &rbfk);	

	for(int idx = 0; idx < train.m_nsamples - 1; idx++) {
		cout << train.y[idx] << "\t";
	}
	cout << train.y[train.m_nsamples - 1] << "\n";
	
	for(int idx = 0; idx < train.m_nsamples - 1; idx++) {
		cout << yt[idx] << "\t";
	}
	cout << yt[train.m_nsamples - 1] << "\n";
	
	int count = 0;
	for(int idx = 0; idx < train.m_nsamples; idx++) {
		if(yt[idx] == train.y[idx]) count++;
	}
	cout << 1.0 * count / train.m_nsamples << "\n";

	svm.clearModel();
	svm.setC(0.25);
	svm.setEps(0.1);
	svm.setSamples(train.m_nsamples);
	svm.setDims(train.m_ndims);
	svm.setTrainData(train.X, train.y);
	t0 = omp_get_wtime();
	svm.train();
	t0 = omp_get_wtime() - t0;
	fprintf(stdout, "elapsed time of svm is %f s\n", t0);

	svm.predict(yt, train.X, train.m_nsamples);	

	for(int idx = 0; idx < train.m_nsamples - 1; idx++) {
		cout << train.y[idx] << "\t";
	}
	cout << train.y[train.m_nsamples - 1] << "\n";
	
	for(int idx = 0; idx < train.m_nsamples - 1; idx++) {
		cout << yt[idx] << "\t";
	}
	cout << yt[train.m_nsamples - 1] << "\n";

	count = 0;	
	for(int idx = 0; idx < train.m_nsamples; idx++) {
		if(yt[idx] == train.y[idx]) count++;
	}
	cout << 1.0 * count / train.m_nsamples << "\n";


//	float * X_test = ArrayAlloc1D<float>(1e5 + 2);
//	for(int i = 0; i < 1e5 + 2; i++) X_test[i] = 1.0;
//	LinearKernel lk;
//	
//	clock_t beg, end;
//	beg = clock();
//	float ret = lk.compute(X_test, X_test, 1e5 + 2);
//	end = clock();
//	cout << (end - beg) << "\n";
//	cout << ret << "\n";
//	cout << test[0] << "\t"
//	     << test[1] << "\t"
//	     << test[2] << "\t"
//	     << test[3] << "\t"
//	     << test[4] << "\t"
//	     << test[5] << "\t"
//	     << test[6] << "\t"
//	     << test[7] << "\n";
	ArrayFree1D<int>(yt);
	yt = nullptr;
	return 0;
}
