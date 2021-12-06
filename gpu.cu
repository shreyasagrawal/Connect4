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
} score_tally_t;

typedef struct board {
    int cells[BOARD_WIDTH * BOARD_HEIGHT];
} board_t;

__host__ __device__ bool board_full(board_t * board) {

    for (int col = 0; col < BOARD_WIDTH; col++) {
        for (int row = 0; row < BOARD_HEIGHT; row++) {
            if (board->cells[col + row * BOARD_WIDTH] == 0) {
                return false;
            }
        }
    }

    return true;
}
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

__host__ __device__ void print_board(board_t * board) {
    printf("-----------------------------\n");
    for (int row = 0; row < BOARD_HEIGHT; row++){ 
        for (int col = 0; col < BOARD_WIDTH; col++) {
            if (board->cells[row * BOARD_WIDTH + col] == 0) {
                printf("|   ");
            } else if (board->cells[row * BOARD_WIDTH + col] == 1) {
                printf("| X ");
            } else {
                printf("| O ");
            }
        }
        printf("|\n");
        printf("-----------------------------\n");
    }
    //Print column numbers for player use
    for (int i = 1; i < 8; i++) {
        printf("| %d ", i);
    }

    printf("\n");
}
__host__ __device__ bool isEmpty(int i) {
    return i == 0;
}
//return difference between computer trees and player threes
//three with zero open ends == 0
//three with one open end == 1
//three with two open ends == 2
__host__ __device__ int three_row (board_t * board) {
    int three_count = 0;
    int ends = 0;
    //Check for consecutive tokens in a given column
    for (int col = 0; col < BOARD_WIDTH; col++) {
        int current = -1;
        int count = 0; // how many consecutive we have found
        for (int row = 0; row < BOARD_HEIGHT; row++) {
            if ((board->cells[col + row * BOARD_WIDTH] == current) && (current != 0)) {
                count++;
                if (count == 3) {
                    //Is top open?
                        //check to make sure the next one in the column is still in bounds
                    if (col + (row + 1) * BOARD_WIDTH < BOARD_WIDTH * BOARD_HEIGHT) {
                        if (isEmpty(board->cells[col + (row + 1) * BOARD_WIDTH])) {
                            //top is open
                            ends++;
                        }
                    }

                    //Is bottom open?
                        //check to make sure the previous one in the column is still in bounds
                    if (col + (row - 1) * BOARD_WIDTH > 0) {
                        if (isEmpty(board->cells[col + (row - 1) * BOARD_WIDTH])) {
                            //bottom is open
                            ends++;
                        }
                    }

                    //If computer 3
                    if (current == 2) {
                        three_count += ends;
                        ends = 0;
                    } else { //case for player 3
                        three_count -= ends;
                        ends = 0;
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
                     //Is right open?
                        //check to make sure the next one in the row is still in the same row
                    if (col + 1 < BOARD_WIDTH) {
                        if (isEmpty(board->cells[col + 1 + row * BOARD_WIDTH])) {
                            //right is open
                            ends++;
                        }
                    }

                    //Is left open?
                        //check to make sure the previous one in the row is still in the same row
                    if (col + 1 > 0) {
                        if (isEmpty(board->cells[col - 1 + row * BOARD_WIDTH])) {
                            //left is open
                            ends++;
                        }
                    }

                    //If computer 3
                    if (current == 2) {
                        three_count += ends;
                        ends = 0;
                    } else { //case for player 3
                        three_count -= ends;
                        ends = 0;
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
    
    if (board_won(board) == 1) {
        return -100;
    }

    if (board_won(board) == 2) {
        return 100;
    }

    int score = 25;
    score += (10 * three_row(board));
    return score;
}

// Take an int, 0-6 to represent the column
// Player parameter (1 for player, 2 for computer)
// Board

__host__ __device__ int play_move(int column, int player, board_t * this_board) {
    if (column < 0 || column > 6) {
        return -1;
    }
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

__global__ void thread_func(board_t * board, score_tally_t * master_counts) {
    //Get the current version of the board for this thread
    
    board_t * local = &board[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK];

    
    //A given block will represent everything for a certain first move
    int first_move = blockIdx.x;

    //Based on the block id, let's (computer) make our first move
    play_move(first_move, 2, local);
    //print_board(board);
    //Check if game is over
    if (board_won(local) == -1) {
        //Player move 1 here
        int second_move = threadIdx.x / (BOARD_WIDTH * BOARD_WIDTH);
        play_move(second_move, 1, local);

        if(board_won(local) == -1) {
            //Computer move 2 here
            int third_move = (threadIdx.x % (BOARD_WIDTH * BOARD_WIDTH)) / 7;
            play_move(third_move, 2, local);
        
            if (board_won(local) ==-1) {
            //Player move 2
            int fourth_move = threadIdx.x % BOARD_WIDTH;
            play_move(fourth_move, 1, local);
            }
        }
    }

    //Step 1 is complete, we have a specific board for this thread

    int this_score = score(local);

    //Update structs based on the score
    
    atomicAdd(&master_counts[blockIdx.x].sum, this_score);
    atomicAdd(&master_counts[blockIdx.x].count, 1);

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

    if (cudaMalloc(&board_gpu, sizeof(board_t) * BOARD_WIDTH * THREADS_PER_BLOCK) != cudaSuccess) {
        fprintf(stderr, "failed to make space on GPU 2");
        exit(2);
    }

    for (int i = 0; i < BOARD_WIDTH * THREADS_PER_BLOCK; i++) {
        if(cudaMemcpy(&board_gpu[i], board, sizeof(board_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "failed to copy memory to GPU");
            exit(2);
        }
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
        double score = (double) master[i].sum / master[i].count;
        if (highest_score < score) { 
            best_move = i;
            highest_score = score;
            //printf("Highest score: %f Best Move: %d\n", highest_score, best_move);
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

    //Welcome player to the game
    printf("Welcome to the game! I'll go first\n");
    //Play the first move
    play_move(3, 2, test);

    int cmp_move = 0;
    int player_move;
    //Repeat this until the game is over
    while (board_won(test) == -1 && !board_full(test)) {
        print_board(test);
        printf("Your move! Enter a number between 1 and 7 to indicate where you would like to drop your token\n");
        //Read in user input
        while (scanf("%d", &player_move) == 0) {
            printf("Please enter a valid number (between 1 and 7)");
        }
        player_move--; // decrement to account for zero indexing
        //Play user move

        while (play_move(player_move, 1, test) == -1) {
            printf("this move is not allowed because there are no more spaces in that column, try again\n");
            while (scanf("%d", &player_move) == 0) {
                printf("Please enter a valid number (between 1 and 7)");
            }
            player_move--;
        }
        printf("\n");

        if (board_won(test) != -1 || board_full(test)) {
            break;
        }
        printf("My turn...\n");
        //Pause

        //Determine the best computer move
        best_move(test, &cmp_move);
        
        //Play the move, unless invalid play, in which case we try other, random moves TODO
        while (play_move(cmp_move, 2, test) == -1) {
            cmp_move = (cmp_move + 1) % BOARD_WIDTH;
        } 
    }

    print_board(test);

    if (board_won(test) == 1) {
        printf("Congratulations! You won\n");
    } else if (board_won(test) == 2) {
        printf("I gotcha there. Better luck next time\n");
    } else {
        printf("Well, we filled it up. Guess we'll call it a draw");
    }

    //TODO; DEAL WITH IO THAT IS NOT NUMBERS, invalid numbers based on board
    
    return 0;
}