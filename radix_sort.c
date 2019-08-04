#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
MPI_Status status;

#define MAX_NUM_BITS 32
int num_of_data = 80000000;

// Generate random data range from 0 to 10000
void random_generate_data(int *arr, int n) {
  for (int i = 0; i < n; i++) {
    arr[i] = rand() % 10000;
  }
}

int main(int argc, char** argv){
  // mpi environment initialization
  int process_id, num_of_process;
  // Init MPI environment
  MPI_Init(&argc, &argv);
  // Identify MPI processes by process_id
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
  // Get the total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_process);
  // Unsorted array
  int *arr = (int*)malloc(num_of_data * sizeof(int));
  int *res = (int*)malloc(num_of_data * sizeof(int));
  // start and end position in array of each process
  int start, end;
  // zero nums and one nums in each process of each bit
  int *num_of_zeros = (int *)malloc(num_of_process * sizeof(int)); 
  int *num_of_ones = (int *)malloc(num_of_process * sizeof(int));
  int *index_of_zeros = (int *)malloc(num_of_process * sizeof(int)); 
  int *index_of_ones = (int *)malloc(num_of_process * sizeof(int));
  struct timeval time_start;
  struct timeval time_end;
  unsigned long dur;
  if(process_id == 0){
    // Random initial the array
    srand(time(0));
    random_generate_data(arr, num_of_data);
    // for (int i = 0; i < num_of_data; i++) {
    //   printf("%d ", arr[i]);
    // }
    // printf("\n");
    // calculate the start and end position of array for each process, only keep in master
    int *start_arr = (int *)malloc(num_of_process * sizeof(int));
    int *end_arr = (int *)malloc(num_of_process * sizeof(int));
    int num = num_of_data / num_of_process;
    int num_remain = num_of_data % num_of_process;
    for (int i = 0; i < num_of_process; i++) {
      if (i == 0) {
        start_arr[0] = 0;
      } else {
        start_arr[i] = end_arr[i - 1] + 1;
      }
      int num_tmp = (i < num_remain) ? (num + 1) : num;
      end_arr[i] = start_arr[i] + num_tmp - 1;
      if (end_arr[i] >= num_of_data) {
        end_arr[i] = num_of_data - 1;
      }
    }
    start = start_arr[0];
    end = end_arr[0];
    gettimeofday(&time_start, NULL);
    // Send the start and end position to each process
    for (int i = 1; i < num_of_process; i++) {
      MPI_Send(&start_arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&end_arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  }
  else{
    MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }
  // Broadcast the array data to all nodes
  MPI_Bcast(arr, num_of_data, MPI_INT, 0, MPI_COMM_WORLD);
  // Calculate the num_of_zeros and num_of_ones
  // of all bits in default array area
  for (int i = 0; i < MAX_NUM_BITS; i++) {
    // Count elements with 0 in i-th bit
    num_of_zeros[process_id] = 0;
    for (int j = start; j <= end; j++) {
      if (((arr[j] >> i) & 1) == 0) {
        num_of_zeros[process_id]++;
      }
    }
    num_of_ones[process_id] = end - start + 1 - num_of_zeros[process_id];
    // Broadcast to update all process array
    for (int j = 0; j < num_of_process; j++) {
      MPI_Bcast(&num_of_zeros[j], 1, MPI_INT, j, MPI_COMM_WORLD);
      MPI_Bcast(&num_of_ones[j], 1, MPI_INT, j, MPI_COMM_WORLD);
    }
    // printf("process_id: %d, num_of_zeros: %d, num_of_ones: %d\n", process_id, num_of_zeros[process_id], num_of_ones[process_id]);
    // Prefix sum to get the zero_index and one_index
    index_of_zeros[process_id] = 0, index_of_ones[process_id] = 0;
    for (int j = 0; j < process_id; j++) {
      index_of_zeros[process_id] += num_of_zeros[j];
      index_of_ones[process_id] += num_of_ones[j];
    }
    // bit '1' should put after bit '0'
    index_of_ones[process_id] += index_of_zeros[process_id];
    for (int j = process_id; j < num_of_process; j++) {
      index_of_ones[process_id] += num_of_zeros[j];
    }
    // printf("process id: %d, index_of_zeros: %d, index_of_ones: %d\n", process_id, index_of_zeros[process_id], index_of_ones[process_id]);
    int ones_tmp = index_of_ones[process_id];
    int zeros_tmp = index_of_zeros[process_id];
    // Move the value to correct position in the array
    for (int j = start; j <= end; j++) {
      if (((arr[j] >> i) & 1) == 0) {
        res[zeros_tmp] = arr[j];
        zeros_tmp++;
      } else {
        res[ones_tmp] = arr[j];
        ones_tmp++;
      }
    }
    // Broadcast the start index of all processes
    for (int j = 0; j < num_of_process; j++) {
      MPI_Bcast(&index_of_zeros[j], 1, MPI_INT, j, MPI_COMM_WORLD);
      MPI_Bcast(&index_of_ones[j], 1, MPI_INT, j, MPI_COMM_WORLD);
    }
    // Write back the result value to arr
    for (int j = index_of_zeros[process_id]; j < zeros_tmp; j++) {
      arr[j] = res[j];
    }
    for (int j = index_of_ones[process_id]; j < ones_tmp; j++) {
      arr[j] = res[j];
    }
    for (int j = 0; j < num_of_process; j++) {
      MPI_Bcast(arr + index_of_zeros[j], num_of_zeros[j], MPI_INT, j,
                MPI_COMM_WORLD);
      MPI_Bcast(arr + index_of_ones[j], num_of_ones[j], MPI_INT, j,
                MPI_COMM_WORLD);
    }
  }
  if (process_id == 0) {
    gettimeofday(&time_end, NULL);
    dur = 1000000 * (time_end.tv_sec - time_start.tv_sec) + time_end.tv_usec - time_start.tv_usec;
    printf("Time for whole using %d processes is %ldus\n", num_of_process, dur);
    // for (int j = 0; j < 50; j++) {
    //   printf("%d ", arr[j]);
    // }
    // printf("\n");
    // for (int j = num_of_data - 50; j < num_of_data; j++) {
    //   printf("%d ", arr[j]);
    // }
    // printf("\n");
  }
  free(arr);
  free(num_of_zeros);
  free(num_of_ones);
  free(index_of_zeros);
  free(index_of_ones);
  free(res);
  MPI_Finalize();
}