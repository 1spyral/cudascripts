#pragma once

#include <cuda_runtime.h>

/**
 * @brief Compares the execution times of regular and atomic increment operations on the GPU.
 * 
 * This function performs a series of tests to compare the execution times of regular and atomic 
 * increment operations on the GPU. It takes arrays specifying the number of threads and elements 
 * for each test, and stores the execution times for both regular and atomic increment operations.
 * 
 * @param num_threads_arr Array containing the number of threads to be used for each test.
 * @param num_elements_arr Array specifying the number of elements to be incremented from 0 in each test.
 * @param count The number of elements in num_threads_arr and num_elements_arr, indicating the number of tests to be performed.
 * @param times_regular Array to store the execution times for regular increment operations in microseconds.
 * @param times_atomic Array to store the execution times for atomic increment operations in microseconds.
 */
void incrementTest(size_t* num_threads_arr, size_t* num_elements_arr, size_t count, int* times_regular, int* times_atomic);