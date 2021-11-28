# Connect4

Noah Fehr, Luke Klein-Collins, Shrey Agrawal

Implementing an AI Connect Four System

What are you planning to build?

We are planning to build a system in which a user can play Connect Four with a computer. The display, based on the worm lab, will allow the user to see the board in a basic user interface as they play the game against the computer.

The basic game of Connect Four which we will be implementing follows these rules: 
Each player is either “X” or “O.” These are analogous to yellow or red in the classic Connect Four game.
Player and computer alternate turns
In a turn, a player may drop their piece into any column. This piece will land on the top of the pieces in that column.
A player wins when there are four of their tiles in a row (horizontally, vertically, or diagonally) 

What are the three concepts your project combines?

GPU: We will use a GPU to compute possible Connect Four boards. In our implementation, the computer will create every possible board over the next n moves, and score each board. Each thread in the GPU will be responsible for one single possible scenario. The number of threads in the GPU will be determined in the implementation, depending on how many moves ahead we are analyzing. 

Threads: Depending on the number of moves the computer analyzes (corresponding directly to the difficulty of the game), we will create a certain number of threads, one for each possible board. For example, if the computer analyzes 1 move ahead, then there are only 7 possible boards that could result from this move, so we will have 7 threads. In the case where the computer analyzes 3 moves ahead, we will create 73 threads, computing each of the possible boards. With a thread for each possible board, we will score every possibility and compile these scores to determine our move. We will have to coordinate between threads and sync threads as we access both possible boards and shared structs which represent the favorability of a given move.

Memory management: With GPU threads writing to memory and storing scores of each possible board, there will be a large amount of data creation, movement, and storage. This will require memory management within our system to store the data produced by these threads and coordinate this access. For example, we foresee implementing a lock to regulate access to the shared structs for each move, which every thread will need to write to at some point. 


How will you implement this project?

From a top level perspective, we are planning to utilize a GPU to create different possible scenarios, determine the favorability of the given scenario, and use the favorability scores of each scenario to determine the best move for the computer. 
The game will begin with the computer playing the first token into the middle column of an empty board. After the user responds with their move, the computer will use the GPU to calculate its next move. At this point, there are seven possible moves (seven columns in which the computer can drop its token) and thus we will want seven structs to store the scores for scenarios moving ahead from that move. For example, one of the structs will keep a running total of the total favorability score for playing a token in Column 0, and the number of possible scenarios analyzed ([running total favorability score, number possible scenarios]). The GPU will use threads to analyze each possible scenario, provided the first move is a token to Column 0, and all of these threads update their corresponding struct, which represents the favorability of their move in a given column. This struct keeps running totals instead of recording every score because we are comparing the average favorability score (total favorability score / number of possible scenarios). After this process, we will have seven structs, each representing a move into that column.
To analyze the implementation, let’s zoom in on one single thread in the GPU. Each specific thread will be responsible for dealing with one of the possible outcomes, given a starting board. This thread will determine the board in this scenario, return the score (a value to indicate how favorable the board is for the computer), and store the score in a struct for that given move. After threads are synced in the GPU, all structs should be updated. Note that because we are using shared structs, we will need to implement some sort of locking mechanisms or controls to coordinate access to these structs. 
After calculating the best score, the computer will complete its best move and wait for the user to respond. After the user makes their next move, the computer will repeat this process. As the game progresses, we will need specific logic to address the cases that arise as the board fills up. Other aspects of the implementation include the function to score a board (given a board scenario, return a ‘favorability score’) and the user interface, which we plan to base on the worm lab provided code. 
To complete this project on time, we are planning to adhere to the following timeline: 
Friday, December 3: GPU set-up complete, with a random scoring algorithm and no user interface
Friday, December 10: Implementation complete, with scoring algorithm and UI
Wednesday, December 15: Presentation complete

What could go wrong with your implementation?

The total number of boards to analyze is contingent on the given board at hand. Players winning the game or attempting to place tokens out of bounds will adjust this number of viable boards. To address these cases, we will put in special cases to compute scores accordingly if a move is on the upper edge of the board or if a move is resulting in a win. 
We also acknowledge the risk that our scoring algorithm does not reflect the best move. If we play a suboptimal move it may result in a possible loss or draw. We believe that we have to accept some errors in this regard. Connect Four is a solved game and if we wanted to write an algorithm to perfectly play we could. However, we want to utilize GPU and so our scoring algorithm may not be perfect but we will try to make it as good as possible.
We will need to ensure atomicity, and other concurrency bugs when multiple threads are manipulating our structs at once. Finally, there will also be over-calculation of the score at some points when we get to 2 identical boards but from different routes/moves.

