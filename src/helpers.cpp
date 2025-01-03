#include <iostream>

void printArray(int* arr, size_t size) {
    if (size == 0) {
        std::cout << "{}";
    } else {
        std::cout << "{ " << arr[0];
        for (size_t i = 1; i < size; i++) {
            std::cout << ", " << arr[i];
        }
        std::cout << " }";
    }
    std::cout << std::endl;
}