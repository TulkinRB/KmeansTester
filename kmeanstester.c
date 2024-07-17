#include <stdio.h>
#include <stdlib.h>

#define INSTRUCTION_FILE_PATH "kmeans_tester/instruction.txt"
#define ALLOC_CALLS_FILE_PATH "kmeans_tester/alloc_calls.txt"

int alloc_to_fail = -2;
int curr_alloc = 0;

void read_alloc_to_fail() {
    FILE *fptr;
    if (alloc_to_fail != -2) {
        return;
    }
    fptr = fopen(INSTRUCTION_FILE_PATH, "r");
    if (fptr == NULL) {
        alloc_to_fail = -1;
        return;
    }
    fscanf(fptr, "%d", &alloc_to_fail);
    fclose(fptr);
}

void write_alloc_call(char type) {
    FILE *fptr = fopen(ALLOC_CALLS_FILE_PATH, "a");
    fputc(type, fptr);
    fclose(fptr);
}

void *tester_malloc(size_t Size) {
    read_alloc_to_fail();
    if (alloc_to_fail == curr_alloc) {
        return NULL;
    }
    else if (alloc_to_fail == -1) {
        write_alloc_call('m');
    }
    curr_alloc++;
    return malloc(Size);
}

void *tester_calloc(size_t NumOfElements, size_t SizeOfElement) {
    read_alloc_to_fail();
    if (alloc_to_fail == curr_alloc) {
        return NULL;
    }
    else if (alloc_to_fail == -1) {
        write_alloc_call('c');
    }
    curr_alloc++;
    return calloc(NumOfElements, SizeOfElement);
}

void *tester_realloc(void *Memory, size_t NewSize) {
    read_alloc_to_fail();
    if (alloc_to_fail == curr_alloc) {
        return NULL;
    }
    else if (alloc_to_fail == -1) {
        write_alloc_call('r');
    }
    curr_alloc++;
    return realloc(Memory, NewSize);
}