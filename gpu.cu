#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>


#define BOARD_WIDTH 7
#define BOARD_HEIGHT 6
#define THREADS_PER_BLOCK 343 //this represents how each block will represent all possible outcomes for the next three moves

typedef struct score_tally { 
    int sum; //running sum of scores
    int count; //count of boards that have been scored
} score_tally_t;

typedef struct board {
    int cells[BOARD_WIDTH * BOARD_HEIGHT];
} board_t;

//Return a boolean; is the board full (ie no possible moves remain)?
__host__ __device__ bool board_full(board_t * board) {
    for (int col = 0; col < BOARD_WIDTH; col++) {
        for (int row = 0; row < BOARD_HEIGHT; row++) {
            if (board->cells[col + row * BOARD_WIDTH] == 0) {
                return false; //if any cell is open, the board is not full
            }
        }
    }
    return true;
}
// Return : 1 if player won, 2 if computer won, -1 if nobody won
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
                //End of consecutive ones, reset current and count based on the cell we are at
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
    //Check for diagonals which run down and right
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

__host__ __device__ void print_board(board_t * board, int rec) {
    //Find index of recent move
    int recent = rec;
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        if (board->cells[i * BOARD_WIDTH + rec] == 0) {
            continue;
        } else {
            recent = i * BOARD_WIDTH + rec;
            break;
        } 
    }
    printf("-----------------------------\n");
    for (int row = 0; row < BOARD_HEIGHT; row++){ 
        for (int col = 0; col < BOARD_WIDTH; col++) {
            if (board->cells[row * BOARD_WIDTH + col] == 0) {
                printf("|   ");
            } else if (board->cells[row * BOARD_WIDTH + col] == 1) {
                if (row * BOARD_WIDTH + col == recent) {
                    //Print most recent move in red for visibility
                    printf("| ");
                    printf("\033[0;31m");
                    printf("X ");
                    printf("\033[0m");
                } else {
                    printf("| X ");
                }  
            } else {
                if (row * BOARD_WIDTH + col == recent) {
                    printf("| ");
                    printf("\033[0;31m");
                    printf("O ");
                    printf("\033[0m");
                } else {
                    printf("| O ");
                } 
            }
        }
        printf("|\n");
        printf("-----------------------------\n");
    }
    printf("-----------------------------\n");
    //Print column numbers for player use
    for (int i = 1; i < 8; i++) {
        printf("| %d ", i);
    }
    printf("\n");
}

//Return a score corresponding the number of three-in-a-rows.
//Player three-in-a-rows are scored as negative, computer as positive
//Points based on how many open ends a three-in-a-row has (0-2) **Note** this has been modified for diagonals
//Used as a helper for the scoring algorithm
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
                    //Is the top in bounds, based on our dimensions?
                    if (col + (row + 1) * BOARD_WIDTH < BOARD_WIDTH * BOARD_HEIGHT) {
                        //Is the top open?
                        if (board->cells[col + (row + 1) * BOARD_WIDTH] == 0) {
                            //Top is open, increment available ends
                            ends++;
                        }
                    }
                    //Note, don't have to check for bottom
                    if (current == 2) { //If computer three in a row
                        three_count += ends;
                        ends = 0;
                    } else { //case for player three in a row
                        three_count -= 2 * ends;
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
                    //Is the right in bounds, based on our dimensions?
                    if (col + 1 < BOARD_WIDTH) {
                        //Is right open?
                        if (board->cells[col + 1 + row * BOARD_WIDTH] == 0) {
                            //right is open
                            ends++;
                        }
                    }
                    //Is the left in bounds, based on our dimensions?
                    if (col - 1 > 0) {
                        //Is left open?
                        if (board->cells[col - 1 + row * BOARD_WIDTH] == 0) {
                            //left is open
                            ends++;
                        }
                    }
                    if (current == 2) { //If computer 3
                        three_count += ends;
                        ends = 0;
                    } else { //case for player 3
                        three_count -= 2 * ends;
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
                    //Check top left corner 
                    int top_left = index - 3 * BOARD_WIDTH - 3;
                    //Check for in bounds and that we are not having wraparounds
                    if (top_left >= 0 && top_left % BOARD_WIDTH < index % BOARD_WIDTH) {
                        if (board->cells[index - 3 * BOARD_WIDTH - 3] == 0) {
                            ends++;
                        }
                    }
                    //Check bottom right
                    int bottom_right = index + BOARD_WIDTH + 1;
                    if (bottom_right < BOARD_WIDTH * BOARD_HEIGHT && bottom_right % BOARD_WIDTH > index % BOARD_WIDTH) {
                        if (board->cells[index + BOARD_WIDTH + 1] == 0) {
                            ends++;
                        }
                    }
                    if (current == 2) {
                        three_count += 2 * ends; //note, we are weighting diagonals more heavily to improve the AI
                        ends = 0;
                    } else {
                        three_count -= 4 * ends; //weighting player diagonal three in a rows as especially bad
                        ends = 0;
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
                    //Check top right corner 
                    int top_right = index - 3 * BOARD_WIDTH + 3;
                    if (top_right >= 0 && index % BOARD_WIDTH < top_right % BOARD_WIDTH) {
                        if (board->cells[index - 3 * BOARD_WIDTH + 3] == 0) {
                            ends++;
                        }
                    }
                    //Check bottom left
                    int bottom_left = index + BOARD_WIDTH - 1;
                    if (bottom_left < BOARD_WIDTH * BOARD_HEIGHT && index % BOARD_WIDTH > bottom_left % BOARD_WIDTH) {
                        if (board->cells[index + BOARD_WIDTH - 1] == 0) {
                            ends++;
                        }
                    }
                    //Check if it's a player or computer 3
                    if (current == 2) {
                        three_count += 2 * ends;
                        ends = 0;
                    } else {
                        three_count -= 4 * ends;
                        ends = 0;
                    }
            } else {
                continue;
            }
        }
    }
    return three_count;
}

//Returns a score representing the favorability of provided board for the computer
__host__ __device__ int score (board_t * board) {
    if (board_won(board) == 1) {
        return -100;
    }
    if (board_won(board) == 2) {
        return 100;
    }
    int score = 50;
    score += (10 * three_row(board));
    return score;
}

// Column parameter: 0 to BOARD_WIDTH - 1 representing where to play the move
// Player parameter : 1 for player, 2 for computer
// Plays the desired move on the given board, return 0 upon success, -1 upon failure
__host__ __device__ int play_move(int column, int player, board_t * this_board) {
    if (column < 0 || column > BOARD_WIDTH - 1) {
        return -1; //if column is out of bounds, return -1 (cannot play this)
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

__global__ void score_board(board_t * board, score_tally_t * master_counts) {
    // Get the current version of the board for this thread
    board_t * local = &board[threadIdx.x + blockIdx.x * THREADS_PER_BLOCK];
    // A given block will represent everything for a certain first move
    int first_move = blockIdx.x;
    // Based on the block id, let's (computer) make our first move
    if (play_move(first_move, 2, local) == -1) {
        return; //trying to play out of bounds
    }
    // Only continue if the board is not won by this move
    if (board_won(local) == -1) {
        // Player's turn to move, based on threadIdx
        int second_move = threadIdx.x / (BOARD_WIDTH * BOARD_WIDTH);
        if (play_move(second_move, 1, local) == -1) {
            return; //trying to play out of bounds
        }
        if(board_won(local) == -1) {
            // Computer's second move, based on threadIdx
            int third_move = (threadIdx.x % (BOARD_WIDTH * BOARD_WIDTH)) / 7;
            if (play_move(third_move, 2, local) == -1) {
                return; //trying to play out of bounds
            }
        
            if (board_won(local) ==-1) {
                // Player's second turn to move, based on threadIdx
                int fourth_move = threadIdx.x % BOARD_WIDTH;
                if (play_move(fourth_move, 1, local) == -1) {
                    return; //trying to play out of bounds
                }
            }
        }
    }
    // Now that we have manipulated the board appropriately for this thread, score the board
    int this_score = score(local);
    // Update structs based on the score, using atomicAdd to avoid concurrency issues
    atomicAdd(&master_counts[blockIdx.x].sum, this_score);
    atomicAdd(&master_counts[blockIdx.x].count, 1);
}

// Call the score_board on GPUs to update structs and determine the best move
void best_move(board_t * board, int * result) {
    // Set up the structs to hold score information
    score_tally_t  master[BOARD_WIDTH];
    // Initialize all structs
    for(int i = 0; i < BOARD_WIDTH; i++){
        master[i].sum = 0;
        master[i].count = 0;
    }
    // GPU Copy and make space
    score_tally_t * master_gpu;
    if (cudaMalloc(&master_gpu, sizeof(score_tally_t) * BOARD_WIDTH) != cudaSuccess) {
        fprintf(stderr, "failed to make space on GPU");
        exit(2);
    }
    if(cudaMemcpy(master_gpu, &master, sizeof(score_tally_t) * BOARD_WIDTH, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "failed to copy memory to GPU");
        exit(2);
    }
    // GPU Copy of board and make space
    board_t * board_gpu;
    // Make space for BOARD_WIDTH * THREADS_PER_BLOCK copies of this, which represents how many threads there are
    // Each thread will need its own copy in order to manipulate the board without affecting other threads
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
    // Run the GPU
    score_board<<<BOARD_WIDTH, THREADS_PER_BLOCK>>>(board_gpu, master_gpu);
    // Wait for kernel to finish
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    // We only have to copy the scoring structs back, not the manipulated boards
    if(cudaMemcpy(&master, master_gpu, sizeof(score_tally_t) * BOARD_WIDTH, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "failed to copy memory from GPU back to CPU");
        exit(2);
    }
    //Free the memory we had allocated on the GPU
    cudaFree(master_gpu);
    cudaFree(board_gpu);
    //Now, look through the array of structs and find the best move
    int best_move = 0;
    double highest_score = -1000; //arbitrary negative number outside of our lowest possible score
    for(int i = 0; i < BOARD_WIDTH; i++){
        double score = (double) master[i].sum / master[i].count;
        if (highest_score < score) { 
            best_move = i;
            highest_score = score;
        }
    }
    //Copy the best move to result so we can access this later
    *result = best_move;
}

int main() {
    //Create a new board and make space
    board_t * test = (board_t*)malloc(sizeof(board_t));
    //Initialize all board values to zero for test
    for (int i = 0; i < BOARD_WIDTH * BOARD_HEIGHT; i++) {
        test->cells[i] = 0;
    }
    //Welcome player to the game
    printf("Welcome to the game! I'll go first\n");
    //Play the first move
    play_move(3, 2, test); //not error checking as the board is initialized to empty, and this is a valid move every time
    int cmp_move = 0;
    int player_move;
    int recent_move = 3; //computer starts by playing the first move in col 3
    //Repeat this until the game is over
    while (board_won(test) == -1 && !board_full(test)) {
        print_board(test, recent_move);
        printf("Your move! Enter a number between 1 and 7 to indicate where you would like to drop your token\n");
        //Read in user input
        int check = scanf("%d", &player_move);
        char junk[500]; //store for user input that is not a valid input (ie not an integer)
        while (check == 0) {
            printf("Please enter a valid number (between 1 and 7)");
            scanf("%s", junk); //scan in any strings, which represent invalid input
            check = scanf("%d", &player_move); //check stores how many values were successfully scanned
        }
        player_move--; // decrement to account for zero indexing
        //Play user move
        while (play_move(player_move, 1, test) == -1) {
            printf("this move is not allowed because there are no more spaces in that column, try again\n");
            check = scanf("%d", &player_move);
            while (check == 0) {
                printf("Please enter a valid number (between 1 and 7)");
                scanf("%s", junk);
                check = scanf("%d", &player_move);
            }
            player_move--;
        }
        printf("\n");
        if (board_won(test) != -1 || board_full(test)) {
            //If the game is over, update recent_move and end while loop
            recent_move = player_move; 
            break;
        }
        printf("My turn...\n");
        //Determine the best computer move
        best_move(test, &cmp_move);
        //Play the move, unless invalid play, in which case we play the next available move
        while (play_move(cmp_move, 2, test) == -1) {
            cmp_move = (cmp_move + 1) % BOARD_WIDTH;
        } 
        recent_move = cmp_move;
    }
    print_board(test, recent_move);
    if (board_won(test) == 1) {
        printf("Congratulations! You won\n");
    } else if (board_won(test) == 2) {
        printf("I gotcha there. Better luck next time\n");
    } else {
        printf("Well, we filled it up. Guess we'll call it a draw");
    }
    free(test);
    return 0;
}