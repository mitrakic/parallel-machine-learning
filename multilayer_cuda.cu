//
//  main.cpp
//  NeuralNetworks
//
//  Created by Pedro Rodriguez on 4/17/14.
//  Copyright (c) 2014 cs189. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <stdlib.h>
#include <cstdlib>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdexcept>
#include <time.h>
#include <cblas.h>
#include "cuda.h"

#include <tbb/tick_count.h>
using namespace std;

int H1SIZE = 300;
int H2SIZE = 100;
int num_digits = 10;
int features = 11;
double ETA = .01;

int chunk_size = 50;
int N = 1279;
int epochs = 100;

void array_print(double* arr, int arrlen) {
    for (int i = 0; i < arrlen; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

__device__ double d_sigmoid(double x) {
     double v = 1.0 / (1.0 + exp(-x));
     return v;
}

__device__ void daxpy(int n, double coef, double* x, double* y) {
    for (int i = 0; i < n; i++) {
     //   double y_1 = y[i];
       double old = atomicAdd(&y[i], x[i] * coef);
//        printf("i %d old %f new %f inc %f \n",i, old, y[i], x[i]*coef);
    }
} 

double sigmoid(double x) {
    double v = 1.0 / (1.0 + exp(-x));
    return v;
}

double mytanh(double x) {
    double v = tanh(x);//(1.0 + tanh(x))/(2.0);
   return v;
}

double mse(double* yk, int label) {
    double sum = 0;
    for (int k = 0; k < num_digits; k++) {
        double tk = (k == label) ? 1.0 : 0.0;
        sum += .5 * pow(yk[k] - tk, 2);
    }
    return sum;
}

double full_mse(double* yk, int* label) {
    double s = 0;
    for (int i = 0; i < N; i++) {
        s += mse(yk + i * num_digits, label[i]);
    }
    return s;
}

void print_output(double* y) {
    for (int i = 0; i < 200; i++) {
        printf("%f\n", y[i]);
    }
}

double eta(int epoch) {
    return 0.001;// / sqrt(epoch + 1);
}

double rand_double() {
    return ((double) rand() / (RAND_MAX))*2 - 1;
}

void print_data_point(double* data_point) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%d ", data_point[i * 28 + j] > 0);
        }
        printf("\n");
    }
}

void print_data_row(double* data_point) {
    for (int i = 0; i < features; i++) {
        printf("%.0f ", data_point[i]);
    }
    printf("\n");
}

void read_all_data(double* x_vals, int* t_vals, char* filename) {
    string line;
    ifstream data(filename);
    if (!data.is_open()) {
        throw invalid_argument("Could not read input file");
    }
    int j = 0;
    int k = 0;
    while (getline(data, line)) {
        istringstream ss(line);
        string token;
        int i = 0;
        while (getline(ss, token, ',')) {
            if (i == features) {
                t_vals[k] = std::stoi(token,nullptr);
                i++;
                k++;
            } else {
                x_vals[j * features + i] = std::stof(token,nullptr);
                i++;
            }
        }
        j++;
    }
    data.close();
    return;
}
double dot(int N, double* X, double* Y){
    double sum = 0;
    for (int i=0;i<N;i++){
        sum+= X[i]*Y[i];
    }
    return sum;
}
__device__ double d_dot(int N, double* X, double* Y){
    double sum = 0;
    for (int i=0;i<N;i++){
        sum+= X[i]*Y[i];
    }
    return sum;
}
void ih1_forward_propagate(double* x, double* h1, double* weights, double* bias, int num_elements) {
    //Dot product x0 with weights and add bk to it.
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < H1SIZE; k++) {
            double result = bias[k] + dot(features, x + i * features, weights + k * features);
            h1[i * H1SIZE + k] = mytanh(result);
        }
    }
    return;
}

__global__ void d_ih1_forward(double* x, double* h1, double* weights, double* bias, int* features,
int* h1size) {
     double result = bias[threadIdx.x] + d_dot(*features, x + (blockIdx.x*(*features)), weights +
(threadIdx.x*(*features)));
     h1[(blockIdx.x*(*h1size))+threadIdx.x] = tanh(result);
}

void h1h2_forward_propagate(double* h1, double* h2, double* weights, double* bias, int num_elements) {
    //Dot product x0 with weights and add bk to it.
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < H2SIZE; k++) {
            double result = bias[k] + dot(H1SIZE, h1 + i * H1SIZE, weights + k * H1SIZE);
            h2[i * H2SIZE + k] = mytanh(result);
        }
    }
    return;
}

__global__ void d_h1h2_forward(double* h1, double* h2, double* weights, double* bias, int*
h1size, int* h2size) {
      double result = bias[threadIdx.x] + d_dot(*h1size, h1 + (blockIdx.x*(*h1size)), weights +
(threadIdx.x*(*h1size)));
      h2[(blockIdx.x*(*h2size))+threadIdx.x] = tanh(result); 
} 

void h2o_forward_propagate(double* h2, double* y, double* weights, double* bias, int num_elements) {
    //Dot product x0 with weights and add bk to it.
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < num_digits; k++) {
            double result = bias[k] + dot(H2SIZE, h2 + i * H2SIZE, weights + k * H2SIZE);
            y[i * num_digits + k] = sigmoid(result);
        }
    }
    return;
}

__global__ void d_h2o_forward(double* h2, double* y, double *weights, double* bias, int* h2size,
int* num_digits) {
     double result = bias[threadIdx.x] + d_dot(*h2size, h2 + (blockIdx.x*(*h2size)), weights +
(threadIdx.x*(*h2size)));
     y[(blockIdx.x*(*num_digits))+threadIdx.x] = d_sigmoid(result); 
}


void ih1_backward_propagate(double* x, double* y,
                            double* delta1, double* weights, double* bias,
                            int num_elements, int epoch) {
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < H1SIZE; k++) {
            double yk = y[i * H1SIZE + k];
            double coef = -1.0  * eta(epoch) * (1 - yk * yk) * delta1[i * H1SIZE + k];
            cblas_daxpy(features, coef, x + i * features, 1, weights + k * features, 1);
            bias[k] += coef;
        }
    }
    return;
}

__global__ void d_ih1_backward(double* x, double* y, double* delta1, double* weights, double* bias, int* h1size, int*
features) {
     double yk = y[(blockIdx.x*(*h1size))+threadIdx.x];
     double coef = -1.0 * 0.001 * (1 - yk * yk) * delta1[(blockIdx.x*(*h1size))+threadIdx.x];
     daxpy(*features, coef, x + (blockIdx.x*(*features)), weights +
(threadIdx.x*(*features)));
     atomicAdd(&bias[threadIdx.x], coef);
} 

void h1h2_backward_propagate(double* x, double* y,
                             double* delta2, double* delta1,
                             double* weights, double* bias, int num_elements, int epoch) {
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < H2SIZE; k++) {
            double yk = y[i * H2SIZE + k];
            double coef = -1.0 * eta(epoch) * (1 - yk * yk) * delta2[i * H2SIZE + k];
            for (int j = 0; j < H1SIZE; j++) {
                delta1[i * H1SIZE + j] += weights[k * H1SIZE + j] * (1 - yk * yk) * delta2[i * H2SIZE + k];
            }
            cblas_daxpy(H1SIZE, coef, x + i * H1SIZE, 1, weights + k * H1SIZE, 1);
            bias[k] += coef;
        }
    }
    return;
}

__global__ void d_h1h2_backward(double* x, double* y, double* delta2, double* delta1, double*
weights, double* bias, int* h2size, int* h1size) {
     double yk = y[(blockIdx.x*(*h2size)) + threadIdx.x];
     double coef = -1.0 * .001 * (1 - yk * yk) * delta2[(blockIdx.x*(*h2size)) + threadIdx.x];
     for (int i = 0; i < *h1size; i++) {
         atomicAdd(&delta1[(blockIdx.x*(*h1size)) + i], weights[(threadIdx.x*(*h1size)) + i] * (1 - yk * yk) * delta2[(blockIdx.x*(*h2size)) + threadIdx.x]); 
     } 
     daxpy(*h1size, coef, x + (blockIdx.x*(*h1size)), weights + (threadIdx.x*(*h1size)));
    // if (blockIdx.x==0) {
    // printf("block %d thread %d yk %f coef %f delta %f\n", blockIdx.x, threadIdx.x, yk, coef, delta2[(blockIdx.x*(*h2size))+threadIdx.x]);
//}     
atomicAdd(&bias[threadIdx.x], coef);
}

void h2o_backward_propagate(double* x, double* y, int* true_vals,
                            double* delta2, double* weights, double* bias,
                            int num_elements, int epoch) {
    double* deltaout = (double*) malloc(num_digits * sizeof(double));
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < num_digits; k++) {
            double yk = y[i * num_digits + k];
            double tk = (k == true_vals[i]) ? 1.0 : 0.0;
            deltaout[k] = yk - tk;//yk * (1 - yk) * (yk - tk);
            double coef = -1.0 * eta(epoch) * deltaout[k];
            for (int j = 0; j < H2SIZE; j++) {
                delta2[i * H2SIZE + j] += weights[k * H2SIZE + j] * deltaout[k];
            }
            cblas_daxpy(H2SIZE, coef, x + i * H2SIZE, 1, weights + k * H2SIZE, 1);
            bias[k] += coef;
        }
    }
    free(deltaout);
    return;
}

__global__ void d_h2o_backward(double* x, double* y, int* true_vals, double* delta2, double*
weights, double* bias, int* h2size, int* num_digits) {
     double tk = (threadIdx.x == true_vals[blockIdx.x]) ? 1.0 : 0.0;
     double yk = y[(blockIdx.x*(*num_digits)) + threadIdx.x];
     double delta = yk - tk;
     double coef = -1.0 * 0.001 * delta;
     //double old;
     //double* lcl_weights = (double* ) malloc(sizeof(double) * *h2size);
     //memcpy(lcl_weights, &weights[threadIdx.x*(*h2size)], sizeof(double)* *h2size); 
     for (int i = 0; i < *h2size; i++) {
         atomicAdd(&delta2[(blockIdx.x*(*h2size)) + i], weights[(threadIdx.x*(*h2size))+i] * delta); 
    //     atomicAdd(&(weights[(threadIdx.x*(*h2size)) + i]), x[(blockIdx.x*(*h2size)) + i]*coef); 
  } 
   //  printf("coef %f\n", coef);
  //   daxpy(*h2size, coef, x + (blockIdx.x*(*h2size)), weights + (threadIdx.x*(*h2size)));
    // atomicAdd(&bias[threadIdx.x], coef);
    //double new_w = weights[threadIdx.x*[ 
//     if (blockIdx.x == 0) {
  //      printf("block: %d thread: %d yk: %f, tk %f, coef %f delta %f\n", blockIdx.x, threadIdx.x, yk, tk, coef, delta); 
  // }
}

__global__ void d_h2o_backward_2(double* x, double* y, int* true_vals, double* weights, double* bias, int* h2size, int* num_digits) {
     double yk = y[(blockIdx.x*(*num_digits)) + threadIdx.x];
     double tk = (threadIdx.x == true_vals[blockIdx.x]) ? 1.0 : 0.0;
     double delta = yk-tk;
     double coef = -1.0 * 0.001 * delta; 
     daxpy(*h2size, coef, x + (blockIdx.x*(*h2size)), weights + (threadIdx.x*(*h2size)));
     atomicAdd(&bias[threadIdx.x], coef);
} 


int classify(double* x, int* output,
             double* ih1weights, double* ih1bias,
             double* h1h2weights, double* h1h2bias,
             double* h2oweights, double* h2obias,
             int num_elements) {
    double* h1 = (double*) malloc(sizeof(double) * H1SIZE * num_elements);
    double* h2 = (double*) malloc(sizeof(double) * H1SIZE * num_elements);
    double* y = (double*) malloc(sizeof(double) * num_digits * num_elements);
    ih1_forward_propagate(x, h1, ih1weights, ih1bias, num_elements);
    h1h2_forward_propagate(h1, h2, h1h2weights, h1h2bias, num_elements);
    h2o_forward_propagate(h2, y, h2oweights, h2obias, num_elements);
    for (int i = 0; i < num_elements; i++) {
        double max = -100.0;
        int argmax = 0;
        for (int k = 0; k < num_digits; k++) {
            if (y[i * num_digits + k] >= max) {
                argmax = k;
                max = y[i * num_digits + k];
            }
        }
        output[i] = argmax;
    }
    free(h1);
    free(h2);
    free(y);
}

double calculate_error(double* x, int num_elements,
                       double* ih1weights, double* ih1bias,
                       double* h1h2weights, double* h1h2bias,
                       double* h2oweights, double* h2obias,
                       int* true_vals) {
    double errors = 0.0;
    int* outputs = (int*) calloc(num_elements, sizeof(int));
    int* classifications = (int*) calloc(num_digits, sizeof(int));
    int* correct_classifications = (int*) calloc(num_digits, sizeof(int));
    classify(x, outputs, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, num_elements);
    for (int i = 0; i < num_elements; i++) {
        classifications[outputs[i]] += 1;
        if (outputs[i] != true_vals[i]) {
            errors += 1;
        } else {
            correct_classifications[outputs[i]] += 1;
        }
    }
    for (int i = 0; i < num_digits; i++) {
         printf("%d: %d/%d\t", i, classifications[i], correct_classifications[i]);
     };
    printf("\n");
    free(outputs);
    free(classifications);
    free(correct_classifications);
    return errors / num_elements;
}

void normalize_data(double* x, int size, double norm) {
    for (int i = 0; i < size; i++) {
        cblas_dscal(features, norm, x + i * features, 1);
    }
}

void shuffle_data(double* x, int* t, int length) {
    //Prepare framework to do random shuffles
    double* x_copy = (double*) malloc(sizeof(double) * length * features);
    int* t_copy = (int*) malloc(sizeof(int) * length);
    vector<int> order;
    for (int i = 0; i < length; ++i) order.push_back(i);
    random_shuffle(order.begin(), order.end());
    int i = 0;
    int k;
    for (std::vector<int>::iterator it=order.begin(); it!=order.end(); ++it) {
        k = *it;
        for (int j = 0; j < features; j++) {
            x_copy[i * features + j] = x[k * features + j];
        }
        t_copy[i] = t[k];
        i++;
    }
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < features; j++) {
            x[i * features + j] = x_copy[i * features + j];
        }
        t[i] = t_copy[i];
    }
    free(x_copy);
    free(t_copy);
}

double finite_difference(int* labels, double* x,
             double* ih1weights, double* ih1bias,
             double* h1h2weights, double* h1h2bias,
             double* h2oweights, double* h2obias) {
    int num_elements = 1;
    double epsilon = .0001;
    double* h1 = (double*) malloc(sizeof(double) * H1SIZE * num_elements);
    double* h2 = (double*) malloc(sizeof(double) * H2SIZE * num_elements);
    double* y = (double*) malloc(sizeof(double) * num_digits * num_elements);

    ih1_forward_propagate(x, h1, ih1weights, ih1bias, num_elements);
    h1h2_forward_propagate(h1, h2, h1h2weights, h1h2bias, num_elements);
    h2o_forward_propagate(h2, y, h2oweights, h2obias, num_elements);
    double mse1 = mse(y, labels[0]);
    double* w = (double*) malloc(sizeof(double) * H1SIZE * features);
    for (int dj = 0; dj < 100; dj++) {
        for (int j = 0; j < H1SIZE * features; j++) {
            w[j] = ih1weights[j];
            if (j == dj) {
                w[j] += epsilon;
            }
        }
        ih1_forward_propagate(x, h1, w, ih1bias, num_elements);
        h1h2_forward_propagate(h1, h2, h1h2weights, h1h2bias, num_elements);
        h2o_forward_propagate(h2, y, h2oweights, h2obias, num_elements);
        double mse2 = mse(y, labels[0]);
        printf("%.20f ", (mse2 - mse1) / epsilon);
        if (dj % 20 == 0) {
            printf("\n");
        }

    }
    free(h1);
    free(h2);
    free(y);
    printf("\n\n");
}

int main(int argc, const char * argv[])
{
    srand (1);

    double* ih1weights = (double*) malloc(sizeof(double) * H1SIZE * features);
    double* ih1bias = (double*) malloc(sizeof(double) * H1SIZE);
    for (int i = 0; i < H1SIZE * features; i++)      ih1weights[i] = rand_double();
    for (int i = 0; i < H1SIZE; i++)            ih1bias[i] = rand_double();


    double* h1h2weights = (double*) malloc(sizeof(double) * H1SIZE * H2SIZE);
    double* h1h2bias = (double*) malloc(sizeof(double) * H2SIZE);
    double* h1delta = (double*) calloc(H1SIZE * chunk_size, sizeof(double));
    for (int i = 0; i < H1SIZE * H2SIZE; i++)      h1h2weights[i] = rand_double();
    for (int i = 0; i < H2SIZE; i++)            h1h2bias[i] = rand_double();


    double* h2oweights = (double*) malloc(sizeof(double) * num_digits * H2SIZE);
    double* h2obias = (double*) malloc(sizeof(double) * num_digits);
    double* h2delta = (double*) calloc(H2SIZE * chunk_size, sizeof(double));

    for (int i = 0; i < H2SIZE * num_digits; i++)      h2oweights[i] = rand_double();
    for (int i = 0; i < num_digits; i++)            h2obias[i] = rand_double();

    int* true_values = (int*) malloc(sizeof(int) * N);
    double* x = (double*) malloc(sizeof(double) * N * features);
    int* label_test = (int*) malloc(sizeof(int) * 320);
    double* x_test = (double*) malloc(sizeof(double) * 320 * features);
    double* h1 = (double*) calloc(chunk_size * H1SIZE, sizeof(double));
    double* h2 = (double*) calloc(chunk_size * H2SIZE, sizeof(double));
    double* y = (double*) malloc(sizeof(double) * chunk_size * num_digits);
    char training_file[] = "./winedata_training_standardized.csv";
    
    read_all_data(x, true_values, training_file);
    
    char test_file[] = "./winedata_test_standardized.csv";
    read_all_data(x_test, label_test, test_file);
    //normalize_data(x, 60000, 1.0/255.0);
    shuffle_data(x, true_values, N);
    //normalize_data(x_test, 320, 1.0/255.0);
    shuffle_data(x_test, label_test, 320);

    double* d_ih1weights;
    double* d_ih1bias;
    double* d_h1h2weights;
    double* d_h1h2bias;
    double* d_h1delta;
    double* d_h2oweights;
    double* d_h2obias;
    double* d_h2delta;

    double* d_x;
    int* d_true_values;
    double* d_h1;
    double* d_h2;
    double* d_y;
    int size = sizeof(double);

    int* d_h1size;
    int* d_h2size;
    int* d_num_digits;
    int* d_features;
    int* d_eta; 

    cudaMalloc((void **) &d_ih1weights, size*H1SIZE*features);
    cudaMalloc((void **) &d_ih1bias, size*H1SIZE);    		
    cudaMalloc((void **) &d_h1h2weights, size*H1SIZE*H2SIZE);    		
    cudaMalloc((void **) &d_h1h2bias, size*H2SIZE);    		
    cudaMalloc((void **) &d_h1delta, size*H1SIZE*chunk_size);    		
    cudaMalloc((void **) &d_h2oweights, size*H2SIZE*num_digits);    		
    cudaMalloc((void **) &d_h2obias, size*num_digits);
    cudaMalloc((void **) &d_h2delta, size*H2SIZE*chunk_size);    		

    cudaMalloc((void **) &d_x, size*N*features);    		
    cudaMalloc((void **) &d_true_values, sizeof(int)*N);    		
    cudaMalloc((void **) &d_h1, size*H1SIZE*chunk_size);    		
    cudaMalloc((void **) &d_h2, size*H2SIZE*chunk_size);    		
    cudaMalloc((void **) &d_y, size*num_digits*chunk_size);    		
    
    cudaMalloc((void **) &d_h1size, sizeof(int));
    cudaMalloc((void **) &d_h2size, sizeof(int));
    cudaMalloc((void **) &d_num_digits, sizeof(int));
    cudaMalloc((void **) &d_features, sizeof(int));
    cudaMalloc((void **) &d_eta, sizeof(int));

    cudaMemcpy(d_h1size, &H1SIZE, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2size, &H2SIZE, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_digits, &num_digits, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_features, &features, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eta, &ETA, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_ih1weights, ih1weights, size*H1SIZE*features, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ih1bias, ih1bias, size*H1SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h1h2weights, h1h2weights, size*H1SIZE*H2SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h1h2bias, h1h2bias, size*H2SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h1delta, h1delta, size*H1SIZE*chunk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2oweights, h2oweights, size*H2SIZE*num_digits, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2obias, h2obias, size*num_digits, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2delta, h2delta, size*H2SIZE*chunk_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, x, size*N*features, cudaMemcpyHostToDevice);
    cudaMemcpy(d_true_values, true_values, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h1, h1, size*H1SIZE*chunk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2, h2, size*H2SIZE*chunk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size*chunk_size*num_digits, cudaMemcpyHostToDevice);
    //printf("h2oweight: %f\n", h2oweights[0]);
    clock_t t1,t2,s1,e1;
    tbb::tick_count tstart = tbb::tick_count::now();
    s1 = clock();
    int passes = 0;
    //finite_difference(true_values, x,
    //             ih1weights, ih1bias,
    //             h1h2weights, h1h2bias,
    //             h2oweights, h2obias);
    //printf("mse=%f\n", full_mse(x, true_values));
    printf("%d,%f,%f\n", 0,
        calculate_error(x, N, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, true_values),
        calculate_error(x_test, 320, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, label_test));
    for (int e = 1; e <= epochs; e++) {
        for (int i = 0; i < N; i += chunk_size) {
            d_ih1_forward<<<chunk_size, H1SIZE>>>(d_x + i * features, d_h1, d_ih1weights, d_ih1bias, d_features,
d_h1size);
cudaMemcpy(h1, d_h1, size*H1SIZE*chunk_size, cudaMemcpyDeviceToHost);
//printf("H1: %f \n", h1[0]);          
            d_h1h2_forward<<<chunk_size, H2SIZE>>>(d_h1, d_h2, d_h1h2weights, d_h1h2bias, d_h1size,
d_h2size);
cudaMemcpy(h2, d_h2, size*H2SIZE*chunk_size, cudaMemcpyDeviceToHost);
//printf("H2 %f \n", h2[0]);
            d_h2o_forward<<<chunk_size, num_digits>>>(d_h2, d_y, d_h2oweights, d_h2obias, d_h2size,
d_num_digits);
cudaMemcpy(y, d_y, size*chunk_size* num_digits, cudaMemcpyDeviceToHost);
//printf("Y: %f \n", y[0]);  
        //   h1h2_forward_propagate(h1, h2, h1h2weights, h1h2bias, chunk_size);
          // h2o_forward_propagate(h2, y, h2oweights, h2obias, chunk_size);
            //h2o_backward_propagate(h2, y, true_values + i, h2delta, h2oweights, h2obias, chunk_size, passes);
            d_h2o_backward<<<chunk_size, num_digits>>>(d_h2, d_y, d_true_values + i, d_h2delta,
d_h2oweights, d_h2obias, d_h2size, d_num_digits);
            d_h2o_backward_2<<<chunk_size, num_digits>>>(d_h2, d_y, d_true_values + i, d_h2oweights, d_h2obias, d_h2size, d_num_digits);
//cudaDeviceSynchronize();
cudaMemcpy(h2oweights, d_h2oweights, size*H2SIZE*num_digits, cudaMemcpyDeviceToHost);
cudaMemcpy(h2obias, d_h2obias, size*num_digits, cudaMemcpyDeviceToHost);
//printf("h2oweights: %f \n", h2oweights[0]);
//printf("h2obias %f\n", h2obias[0]);                        
// h1h2_backward_propagate(h1, h2, h2delta, h1delta, h1h2weights, h1h2bias, chunk_size, passes);
           d_h1h2_backward<<<chunk_size, H2SIZE>>>(d_h1, d_h2, d_h2delta, d_h1delta,
d_h1h2weights, d_h1h2bias, d_h2size, d_h1size);
cudaMemcpy(h1h2weights, d_h1h2weights, size*H1SIZE*H2SIZE, cudaMemcpyDeviceToHost);
cudaMemcpy(h1h2bias, d_h1h2bias, size*H2SIZE, cudaMemcpyDeviceToHost);
//printf("h1h2weight %f, h1h2bias %f\n", h1h2weights[0], h1h2bias[0]); 
            d_ih1_backward<<<chunk_size, H1SIZE>>>(d_x + i * features, d_h1, d_h1delta,
d_ih1weights, d_ih1bias, d_features,
d_h1size);            
           // memset(h1delta, 0, sizeof(double) * H1SIZE * chunk_size);
           // memset(h2delta, 0, sizeof(double) * H2SIZE * chunk_size);
           cudaMemcpy(d_h1delta, h1delta, size*H1SIZE*chunk_size, cudaMemcpyHostToDevice);
           cudaMemcpy(d_h2delta, h2delta, size*H2SIZE*chunk_size, cudaMemcpyHostToDevice);
           cudaMemcpy(d_h1, h1, size*H1SIZE*chunk_size, cudaMemcpyHostToDevice);
           cudaMemcpy(d_h2, h2, size*H2SIZE*chunk_size, cudaMemcpyHostToDevice);
           // memset(h1, 0, sizeof(double) * H1SIZE * chunk_size);
           // memset(h2, 0, sizeof(double) * H2SIZE * chunk_size);
            passes += 1;
        }
        shuffle_data(x, true_values, N);
        e1 = clock();
        double es = (double) (e1 - s1) / CLOCKS_PER_SEC;
        s1 = clock();
        cudaMemcpy(ih1weights, d_ih1weights, size*H1SIZE*features, cudaMemcpyDeviceToHost);
        cudaMemcpy(ih1bias, d_ih1bias, size*H1SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h1h2weights, d_h1h2weights, size*H2SIZE*H1SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h1h2bias, d_h1h2bias, size*H2SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2oweights, d_h2oweights, size*H2SIZE*num_digits, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2obias, d_h2obias, size*num_digits, cudaMemcpyDeviceToHost);
  //      printf("%f %f\n", h2obias[0], h2oweights[0]); 
        printf("%d,%f,%f\n", e,
            calculate_error(x, N, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, true_values),
            calculate_error(x_test, 320, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, label_test));
    }
    t2 = clock();
    tbb::tick_count tend = tbb::tick_count::now();
    double diff = (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("Running time: %fs\n", (tend-tstart).seconds());
    
    free(x);
    free(ih1weights);
    free(ih1bias);
    free(h1h2weights);
    free(h1h2bias);
    free(h2oweights);
    free(h2obias);
    free(h1delta);
    free(h2delta);
    free(true_values);
    free(h1);
    free(h2);
    free(y);

    cudaFree(d_x);
    cudaFree(d_true_values);
    cudaFree(d_h1);
    cudaFree(d_h2);
    cudaFree(d_y);

    cudaFree(d_ih1weights);
    cudaFree(d_ih1bias);
    cudaFree(d_h1h2weights);
    cudaFree(d_h1h2bias);
    cudaFree(d_h2oweights);
    cudaFree(d_h2obias);
    cudaFree(d_h1delta);
    cudaFree(d_h2delta);

    cudaFree(d_h1size);
    cudaFree(d_h2size);
    cudaFree(d_num_digits);
    cudaFree(d_features);
    cudaFree(d_eta);
    return 0;
}
