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
#include <omp.h>
#include <time.h>
#include <cblas.h>
using namespace std;

int H1SIZE = 300;
int H2SIZE = 100;
int num_digits = 10;
int features = 784;
double ETA = .01;


int chunk_size = 200;
int N = 50000;
int epochs = 600;

void array_print(double* arr, int arrlen) {
    for (int i = 0; i < arrlen; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
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
            char token_array[4];
            token_array[0] = token[0];
            token_array[1] = token[1];
            token_array[2] = token[2];
            token_array[3] = '\0';
            if (i == features) {
                t_vals[k] = atoi(token_array);
                i++;
                k++;
            } else {
                x_vals[j * features + i] = atof(token_array);
                i++;
            }
        }
        j++;
    }
    data.close();
    return;
}

void ih1_forward_propagate(double* x, double* h1, double* weights, double* bias, int num_elements) {
    //Dot product x0 with weights and add bk to it.
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < H1SIZE; k++) {
            double result = bias[k] + cblas_ddot(features, x + i * features, 1, weights + k * features, 1);
            h1[i * H1SIZE + k] = mytanh(result);
        }
    }
    return;
}

void h1h2_forward_propagate(double* h1, double* h2, double* weights, double* bias, int num_elements) {
    //Dot product x0 with weights and add bk to it.
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < H2SIZE; k++) {
            double result = bias[k] + cblas_ddot(H1SIZE, h1 + i * H1SIZE, 1, weights + k * H1SIZE, 1);
            h2[i * H2SIZE + k] = mytanh(result);
        }
    }
    return;
}

void h2o_forward_propagate(double* h2, double* y, double* weights, double* bias, int num_elements) {
    //Dot product x0 with weights and add bk to it.
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < num_digits; k++) {
            double result = bias[k] + cblas_ddot(H2SIZE, h2 + i * H2SIZE, 1, weights + k * H2SIZE, 1);
            y[i * num_digits + k] = sigmoid(result);
        }
    }
    return;
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
    // for (int i = 0; i < num_digits; i++) {
    //     printf("%d: %d/%d\t", i, classifications[i], correct_classifications[i]);
    // };
    //printf("\n");
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

    int* true_values = (int*) malloc(sizeof(int) * 60000);
    double* x = (double*) malloc(sizeof(double) * 60000 * features);

    double* h1 = (double*) calloc(chunk_size * H1SIZE, sizeof(double));
    double* h2 = (double*) calloc(chunk_size * H2SIZE, sizeof(double));
    double* y = (double*) malloc(sizeof(double) * chunk_size * num_digits);
    char training_file[] = "../train-full.txt";
    read_all_data(x, true_values, training_file);
    normalize_data(x, 60000, 1.0/255.0);
    shuffle_data(x, true_values, 60000);
    double* x_test = x + features * 50000;
    int* label_test = true_values + 50000;
    clock_t t1,t2,s1,e1;
    t1 = clock();
    s1 = clock();
    int passes = 0;
    //finite_difference(true_values, x,
    //             ih1weights, ih1bias,
    //             h1h2weights, h1h2bias,
    //             h2oweights, h2obias);
    //printf("mse=%f\n", full_mse(x, true_values));
    printf("%d,%f,%f\n", 0,
        calculate_error(x, N, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, true_values),
        calculate_error(x_test, 10000, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, label_test));
    for (int e = 1; e <= epochs; e++) {
        for (int i = 0; i < N; i += chunk_size) {
            ih1_forward_propagate(x + i * features, h1, ih1weights, ih1bias, chunk_size);
            h1h2_forward_propagate(h1, h2, h1h2weights, h1h2bias, chunk_size);
            h2o_forward_propagate(h2, y, h2oweights, h2obias, chunk_size);
            h2o_backward_propagate(h2, y, true_values + i, h2delta, h2oweights, h2obias, chunk_size, passes);
            h1h2_backward_propagate(h1, h2, h2delta, h1delta, h1h2weights, h1h2bias, chunk_size, passes);
            ih1_backward_propagate(x + i * features, h1, h1delta, ih1weights, ih1bias, chunk_size, passes);
            memset(h1delta, 0, sizeof(double) * H1SIZE * chunk_size);
            memset(h2delta, 0, sizeof(double) * H2SIZE * chunk_size);
            memset(h1, 0, sizeof(double) * H1SIZE * chunk_size);
            memset(h2, 0, sizeof(double) * H2SIZE * chunk_size);
            passes += 1;
        }
        shuffle_data(x, true_values, N);
        e1 = clock();
        double es = (double) (e1 - s1) / CLOCKS_PER_SEC;
        s1 = clock();
        printf("%d,%f,%f\n", e,
            calculate_error(x, N, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, true_values),
            calculate_error(x_test, 10000, ih1weights, ih1bias, h1h2weights, h1h2bias, h2oweights, h2obias, label_test));
    }
    t2 = clock();
    double diff = (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("Running time: %fs\n", diff);
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
    return 0;
}
