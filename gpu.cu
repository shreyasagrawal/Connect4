#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>


#define BOARD_WIDTH 7
#define BOARD_HEIGHT 6
#define THREADS_PER_BLOCK 343

typedef struct score_tally { 
    int sum;
    int count; //count of total scored boards
    //Lock lock; //lock to avoid multiple threads writing at once
} score_tally_t;

typedef struct board {
    int cells[BOARD_WIDTH * BOARD_HEIGHT];
} board_t;

// 1 if player won, 2 if computer 1, -1 if nobody won
__host__ __device__ int board_won(board_t * board) {
    //Check for consecutive tokens in a given column
    for (int col = 0; col < BOARD_WIDTH; col++) {
        int current = -1;
        int count = 0; // how many consecutive we have found
        for (int row = 0; row < BOARD_HEIGHT; row++) {
            if ((board->cells[col + row * BOARD_WIDTH] == current) && (current != 0)) {
                count++;
                if (count == 4) {
                    return board->cells[col + row * BOARD_WIDTH];
                }

            } else {
                current = board->cells[col + row * BOARD_WIDTH];
                count = 1;
            }
        }
    }

    //Check for consecutive tokens in a given row
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        int current = -1;
        int count = 0;
        for (int col = 0; col < BOARD_WIDTH; col++) {

            if ((board->cells[col + row * BOARD_WIDTH] == current) && (current != 0)) {
                count++;
                if (count == 4) {
                    return board->cells[col + row * BOARD_WIDTH];
                }

            } else {
                current = board->cells[col + row * BOARD_WIDTH];
                count = 1;
            }
        }
    }

    //Check down rights
    for (int col = 0; col < 4; col++) {
        //Check cells until there are three rows remaining
        for (int row = 0; row < BOARD_HEIGHT - 3; row++) {
            int current = board->cells[col + row * BOARD_WIDTH];
            int count = 1;
            if (current == 0) {
                continue; //not interested
            }
            //Check the three cells down and right
            int index = col + row * BOARD_WIDTH;
            for (int i = 0; i < 3; i++) {
                index += BOARD_WIDTH + 1;
                if (board->cells[index] == current) {
                    count++;
                    continue;
                } else {
                    break;
                }
            }

            if (count == 4) {
                return board->cells[col + row * BOARD_WIDTH];
            } else {
                continue;
            }
        }
    }

    //Check down lefts
    for (int col = 3; col < 7; col++) {
        //Check cells until there are three rows remaining
        for (int row = 0; row < BOARD_HEIGHT - 3; row++) {
            int current = board->cells[col + row * BOARD_WIDTH];
            int count = 1;
            if (current == 0) {
                continue; //not interested
            }
            //Check the three cells down and left
            int index = col + row * BOARD_WIDTH;
            for (int i = 0; i < 3; i++) {
                index += BOARD_WIDTH - 1;
                if (board->cells[index] == current) {
                    count++;
                    continue;
                } else {
                    break;
                }
            }

            if (count == 4) {
                return board->cells[col + row * BOARD_WIDTH];
            } else {
                continue;
            }
        }
    }

    return -1; 
}

void print_board(board_t * board) {
    printf("-----------------------------\n");
    for (int row = 0; row < BOARD_HEIGHT; row++){ 
        for (int col = 0; col < BOARD_WIDTH; col++) {
            printf("| %d ", board->cells[row * BOARD_WIDTH + col]);
        }
        printf("|\n");
        printf("-----------------------------\n");
    }
}

//return difference between computer trees and player threes
__host__ __device__ int three_row (board_t * board) {
    int three_count = 0;

    //Check for consecutive tokens in a given column
    for (int col = 0; col < BOARD_WIDTH; col++) {
        int current = -1;
        int count = 0; // how many consecutive we have found
        for (int row = 0; row < BOARD_HEIGHT; row++) {
            if ((board->cells[col + row * BOARD_WIDTH] == current) && (current != 0)) {
                count++;
                if (count == 3) {
                    //Check if it's a player or computer 3
                    if (current == 2) {
                        three_count++;
                    } else {
                        three_count--;
                    }
                }

            } else {
                current = board->cells[col + row * BOARD_WIDTH];
                count = 1;
            }
        }
    }

    //Check for consecutive tokens in a given row
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        int current = -1;
        int count = 0;
        for (int col = 0; col < BOARD_WIDTH; col++) {

            if ((board->cells[col + row * BOARD_WIDTH] == current) && (current != 0)) {
                count++;
                if (count == 3) {
                    //Check if it's a player or computer 3
                    if (current == 2) {
                        three_count++;
                    } else {
                        three_count--;
                    }
                }

            } else {
                current = board->cells[col + row * BOARD_WIDTH];
                count = 1;
            }
        }
    }

    //Check down rights
    for (int col = 0; col < 5; col++) {
        //Check cells until there are three rows remaining
        for (int row = 0; row < BOARD_HEIGHT - 2; row++) {
            int current = board->cells[col + row * BOARD_WIDTH];
            int count = 1;
            if (current == 0) {
                continue; //not interested
            }
            //Check the three cells down and right
            int index = col + row * BOARD_WIDTH;
            for (int i = 0; i < 4; i++) {
                index += BOARD_WIDTH + 1;
                if (board->cells[index] == current) {
                    count++;
                    continue;
                } else {
                    break;
                }
            }

            if (count == 3) {
                    //Check if it's a player or computer 3
                    if (current == 2) {
                        three_count++;
                    } else {
                        three_count--;
                    }
            } else {
                continue;
            }
        }
    }

    //Check down lefts
    for (int col = 2; col < 7; col++) {
        //Check cells until there are three rows remaining
        for (int row = 0; row < BOARD_HEIGHT - 2; row++) {
            int current = board->cells[col + row * BOARD_WIDTH];
            int count = 1;
            if (current == 0) {
                continue; //not interested
            }
            //Check the three cells down and left
            int index = col + row * BOARD_WIDTH;
            for (int i = 0; i < 4; i++) {
                index += BOARD_WIDTH - 1;
                if (board->cells[index] == current) {
                    count++;
                    continue;
                } else {
                    break;
                }
            }

            if (count == 3) {
                    //Check if it's a player or computer 3
                    if (current == 2) {
                        three_count++;
                    } else {
                        three_count--;
                    }
            } else {
                continue;
            }
        }
    }

    return three_count;
}

__host__ __device__ int score (board_t * board) {
    
    if (board_won(board) == 2) {
        return 100;
    }

    if (board_won(board) == 1) {
        return 0;
    }

    int score = 0;

    //Three in a row (0, 1, 2 ends open) 0 2 2 0 0
    return score;
}

// Take an int, 0-6 to represent the column
// Player parameter (1 for player, 2 for computer)
// Board

__host__ __device__ int play_move(int column, int player, board_t * this_board) {
    //Find the next available spot in a given column
    for (int row = BOARD_HEIGHT - 1; row >= 0; row--) {
        if (this_board->cells[row * BOARD_WIDTH + column] == 0) {
            this_board->cells[row * BOARD_WIDTH + column] = player;
            return 0;
        }
    }

    //If we don't find any free spots, this is an invalid move. Return an error val
    return -1;
}
//Given a board, (1) update the board to reflect a specific situation (2) "score" the board (3) update structs
//Block > 1024 threads

//Number of blocks is always divisble by 7 --
//each "move" will be assigned a certain number of blocks, based on the number of moves to think ahead


//BlockID will dictate the first move
//Now, there are 7^3 possbilities
//0-343 correspond to boards
// 343 == 7 * 7 * 7
//49 possible
//player move 1 * 49 + comp move 1 * 7 + player move 2
//threadIdx / 49 == player move 1
//(threadIdx % 49) / 7 == comp move 1
//threadIdx % 7 == player move 2

__global__ void thread_func(board_t * board, score_tally_t * master_counts) {
    
    //A given block will represent everything for a certain first move
    int first_move = blockIdx.x;

    //Based on the block id, let's (computer) make our first move
    play_move(first_move, 2, board);

    //Check if game is over
    if (board_won(board) == -1) {
        //Player move 1 here
        int second_move = threadIdx.x / (BOARD_WIDTH * BOARD_WIDTH);
        play_move(second_move, 1, board);

        if(board_won(board) == -1) {
            //Computer move 2 here
            int third_move = (threadIdx.x % (BOARD_WIDTH * BOARD_WIDTH)) / 7;
            play_move(third_move, 2, board);
        
            if (board_won(board) ==-1) {
            //Player move 2
            int fourth_move = threadIdx.x % BOARD_WIDTH;
            play_move(fourth_move, 1, board);
            }
        }
    }

    //Step 1 is complete, we have a specific board for this thread

    int this_score = score(board);

    //Update structs based on the score
    
    atomicAdd(&master_counts[blockIdx.x].sum, this_score);
    atomicAdd(&master_counts[blockIdx.x].count, 1);
    // pthread_mutex_lock(&master_counts[blockIdx.x].lock);
    // //now, we have the lock, update struct

    // master_counts[blockIdx.x].sum += this_score;
    // master_counts[blockIdx.x].count++;

    // pthread_mutex_unlock(&master_counts[blockIdx.x].lock);

}

//Returns something between 0 and 111
int borders(int current) {
    int borders = 0;
    if (current % BOARD_WIDTH - 1 < 0) {
        //We are at the left border
        borders += 1;
    }
    if (current % BOARD_WIDTH + 1 == BOARD_WIDTH) {
        //We are at the right border
        borders += 10;
    }
    if (current + BOARD_WIDTH > BOARD_WIDTH * BOARD_HEIGHT) {
        //We are at the bottom border
        borders += 100;
    }
    return borders;
}



void best_move(board_t * board, int * result) {
    //Set up the struct
    score_tally_t  master[BOARD_WIDTH];

    //Initialize all structs
    for(int i = 0; i < BOARD_WIDTH; i++){
        master[i].sum = 0;
        master[i].count = 0;
        //master[i].lock = PTHREAD_MUTEX_INITIALIZER;
    }

    //GPU Copy and make space
    score_tally_t * master_gpu;

    if (cudaMalloc(&master_gpu, sizeof(score_tally_t) * BOARD_WIDTH) != cudaSuccess) {
        fprintf(stderr, "failed to make space on GPU");
        exit(2);
    }

    if(cudaMemcpy(master_gpu, &master, sizeof(score_tally_t) * BOARD_WIDTH, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "failed to copy memory to GPU");
        exit(2);
    }

    //GPU Copy of board and make space
    board_t * board_gpu;

    if (cudaMalloc(&board_gpu, sizeof(board_t)) != cudaSuccess) {
        fprintf(stderr, "failed to make space on GPU 2");
        exit(2);
    }

    if(cudaMemcpy(board_gpu, board, sizeof(board_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "failed to copy memory to GPU");
        exit(2);
    }

    thread_func<<<BOARD_WIDTH, THREADS_PER_BLOCK>>>(board_gpu, master_gpu);

    //Wait for kernel to finish
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }

    if(cudaMemcpy(&master, master_gpu, sizeof(score_tally_t) * BOARD_WIDTH, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "failed to copy memory from GPU back to CPU");
        exit(2);
    }

    cudaFree(master_gpu);
    cudaFree(board_gpu);

    //Now, look through the array of structs and find the best move
    int best_move = 0;
    double highest_score = 0;
    for(int i = 0; i < BOARD_WIDTH; i++){
        if (highest_score < (master[i].sum / master[i].count)) {
            best_move = i;
            highest_score = master[i].sum / master[i].count;
        }
    }

    *result = best_move;
}

int main() {
    board_t * test = (board_t*)malloc(sizeof(board_t));
    //Initialize all board values to zero for test
    for (int i = 0; i < BOARD_WIDTH * BOARD_HEIGHT; i++) {
        test->cells[i] = 0;
    }

    play_move(3, 2, test);
    play_move(3, 2, test);
    play_move(3, 2, test);
    if (three_row(test) > 0) {
        play_move(3,1,test);
    }

    play_move(2, 1, test);

    int next_move = 3;
    best_move(test, &next_move);

    play_move(next_move, 2, test);

    // while(board_won(test) == -1) {
    //     if (play_move(3, 1, test) == -1) {
    //         break;
    //     }
    // }
    

    print_board(test);
    return 0;
}