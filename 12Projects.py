#12 Projects in python
 
#1) MADLIBS 1
#Logic - we want to make the sentences and varibales with a input value together so conactenation of them .
#String Concatenation 
# we want to create a fill up sort para 
# Just Example of variable  -Youtuber="Ishika Chhoker"#made a string with a value 
# How to conctenate the variable to the string -print(f"subscribe to {Youtuber}")#cocatenated it , another way is print("subscribe to"+Youtuber)
adj1=input("adjective ")
verb1=input('enter verb')
verb2=input('again enter verb')
Famperson=input("FamousPerson:")
#use f and curly brackets to bring a string and a input value or var together
madlib=f"Computer Programming is so {adj1}.It makes me excited about the time because I love to \
    {verb1} . Stay hydrated and {verb2}, like you are {Famperson}."
print(madlib)


#2 )GUESS THE NUMBER(COMPUTER)
#Logic - we just used lib to get the computer have a random no in a var stored , and then we will loop and give multiple chanecs for it to guess 
import random #first we imported the random lib
def guess(x):#function is guess with range
    randomNumber=random.randint(1,x) #it will choose the random number in range 
    guess=0 #initialise variable guess 
    while guess!=randomNumber:
        guess=int(input(f'enter your guess for our number btw 1 to{x} '))
        if guess<randomNumber:
            print('too low')
        elif guess>randomNumber:
            print('too high')

    print(f'yay you guessed correctly {randomNumber} ')        

guess(20)

#3)GUESS THE NUMBER (USER)
#logic - two pointers , and feedbacks , then update ptr as per feedbacks then make while and if else loops 
import random
def computer_guess(x):
    low=1
    high=x 
    feedback=''
    while feedback!='c':
        guess=random.randint(low,high)
        print(f'the guess is {guess}')
        feedback=input('give the feedback c , l or h-')
        if feedback=='l':
            low=guess+1
        elif feedback=='h':
            high=guess-1

    print(f'you won hurray  {guess}')        
               
computer_guess(10)  


#4)ROCK PAPER SCISSORS 
#logic - first craete a fn in it compared the both status then define winning condition in other fn and then calling it in the above one then defining in what win what to be said 
import random 
def play():
    user=input("Choose any value r for rock , p for paper and s for scissors:")
    computer=random.choice(['r','p','s'])
    if user==computer:
        return'its a tie'

    if iswin(user, computer):
        return 'shit u won'
    if iswin(computer, user):
        return 'haha i won '        


def iswin(player, opponent):
    if player=='r'and opponent=='s' or player=='s'and opponent=='p'or player=='p'and opponent =='r':
        return True


print(play())

#5) HANGMAN
#logic- find a valid word then create set of alphabets , used , user letter(current chosen ), word letters , then check if they exists where and remove and join and display
 
import random    
from HangmanWords import words
import string      #set used

def validword(words):     #valid words are being find as some have blanks nd things 
    word = random.choice(words)
    while '-' in word or ' ' in word:
        word = random.choice(words)
    return word.upper()            #all in caps 


def hangman():
    word = validword(words)       #chose a word 
    word_letters = set(word)      #collect the word letters
    alphabet = set(string.ascii_uppercase)      #collect all alphabets in caps
    used_letters = set()           #set of all used char
    
    while len(word_letters) > 0:     #run till no of letter in word is zero
        print('You have used these letters:', ' '.join(used_letters))     #show all used letters    
        word_list = [letter if letter in used_letters else '-' for letter in word]   #if letter is guessed then add it else put a blank
        print('Current word is:', ' '.join(word_list))     #show the current word with blanks 

        user_letter = input("Guess a letter: ").upper()       #guess a letter  , takea letter ip
        if user_letter in alphabet - used_letters:            #if users letter is a new letter
            used_letters.add(user_letter)                    #add it to used ones 
            if user_letter in word_letters:                  #if its in word 
                word_letters.remove(user_letter)              #remove that letter from word 
        elif user_letter in used_letters:                        
            print("You have already guessed it.")
        else:
            print("Invalid character, try again.")  # Fixed invalid char message

    print(f"Congratulations! The word was {word}")

hangman()


#6 and #7)TIC TAC TOE +AI

from TicTacToePlayer import HumanPlayer, RandomComputerPlayer
class TicTacToe:
    def __init__(self):
        self.board=[' 'for _ in range(9)]#we will use single list to represent 3 x 3 board 
        self.current_wimnner=None #keep track of the winner 

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range (3)]: #this is just getting the rows 
            print('| '+'|'.join(row)+'|') #statment 1 

    @staticmethod
    def print_board_nums():
       # giving indexes to each print board block or box , think like a matrix
       # 0|1|2 etc (tells us what number corresponds to what box )
        number_board=[[str(i)for i in range(j*3, (j+1)*3)]for j in range(3)]        
        for row in number_board:
             print('| '+'|'.join(row)+'|') #just copy statement 1
    #we are representing emplty space with a space 
    def available_moves(self):
        return  [i for i , spot in enumerate (self.board)if spot ==' ']
        #moves=[]
        #for (i,x) in enumerate (self.board):
            #['x','x','o', ]---> [(0,'x'),(1,'x'),(2,'o')]     
            #if spot == ' ':
            #   moves.append(i)
            #return moves
    def empty_squares(self):
        return ' ' in self.board
    def num_empty_squares(self):
        return self.board.count(' ')
    def make_move(self, square, letter):
        #if valid move then make a move assign square to letter
        #then return true ,if invalid , return false
        if self.board[square]==' ':
            self.board[square]=letter
            if self.winner(square, letter):
                self.current_winner= letter
            return True 
        return False
    def winner(self, square, letter):
        #winner if 3 in a row anywhree we have to check all of thse 
        #first lets check the row 
        row_ind= square//3 
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all ([spot == letter for spot in row]):
            return True 
        #check colums 
        col_ind= square% 3
        column = [self.board[col_ind +i *3 ]for i in range (3)]
        if all ([spot == letter for spot in column ]):
            return True 
        #check diagonals 
        # but only if the square is an even number (0,2,4,6,8)
        #these are the only moves possible to win a diagonal 
        if square %2 ==0:
            diagonal1= [self.board[i] for i in [0,4,8]]#left to right diagonal 
            if all ([spot==letter for spot in diagonal1]):
                return True
            diagonal2=[self.board[i] for i in [2,4,6]]#right to left diagonal 
            if all ([spot ==letter for spot in diagonal2]):
                return True 
        #if all thse fail 
        return False
        
        

def play (game, x_player, o_player, print_game= True  ):
    #returns the winner of the game (the letter ) or none for a tie 
    if print_game:
        game.print_board_nums()

    letter ='X'#starting letter 
    #iterate while the games still has empty sqyuares 
    # we dont have to worry about winner cuz we ll just return that 
    # which breaks the loop
    while game.empty_squares(): 
        #get the move from the appropriate player 
        if letter =='O':
            square = o_player.get_move(game )
        else :
            square = x_player.get_move(game )

        #lets define a function to make a move 
        if game.make_move(square, letter):
            if print_game:
                print(letter + f'makes a move to square {square }')
                game.print_board()
                print('')#just empty line 
            if game.current_winner:
                if print_game:
                    print(letter+'wins!')
                    return letter    
            #after we made our move we need to alternate letters 
            letter = 'O' if letter == 'X'else 'X'#switches player 
            #if letter =='X'
            #    letter ='O' 
            # else :
            # letter = 'X'
            
            #BUT WHAT IF WE WON ??? 
            if print_game:
                print('its a tie ')  

if __name__=='__main__':
    x_player =HumanPlayer('X')
    o_player= RandomComputerPlayer('O')
    t = TicTacToe()
    play(t, x_player, o_player, print_game=True) 



#8)MINESWEEPER

import random
import re

# lets create a board object to represent the minesweeper game
# this is so that we can just say "create a new board object", or
# "dig here", or "render this game for this object"
class Board:
    def __init__(self, dim_size, num_bombs):
        # let's keep track of these parameters. they'll be helpful later
        self.dim_size = dim_size
        self.num_bombs = num_bombs

        # let's create the board
        # helper function!
        self.board = self.make_new_board() # plant the bombs
        self.assign_values_to_board()

        # initialize a set to keep track of which locations we've uncovered
        # we'll save (row,col) tuples into this set 
        self.dug = set() # if we dig at 0, 0, then self.dug = {(0,0)}

    def make_new_board(self):
        # construct a new board based on the dim size and num bombs
        # we should construct the list of lists here (or whatever representation you prefer,
        # but since we have a 2-D board, list of lists is most natural)

        # generate a new board
        board = [[None for _ in range(self.dim_size)] for _ in range(self.dim_size)]
        # this creates an array like this:
        # [[None, None, ..., None],
        #  [None, None, ..., None],
        #  [...                  ],
        #  [None, None, ..., None]]
        # we can see how this represents a board!

        # plant the bombs
        bombs_planted = 0
        while bombs_planted < self.num_bombs:
            loc = random.randint(0, self.dim_size**2 - 1) # return a random integer N such that a <= N <= b
            row = loc // self.dim_size  # we want the number of times dim_size goes into loc to tell us what row to look at
            col = loc % self.dim_size  # we want the remainder to tell us what index in that row to look at

            if board[row][col] == '*':
                # this means we've actually planted a bomb there already so keep going
                continue

            board[row][col] = '*' # plant the bomb
            bombs_planted += 1

        return board

    def assign_values_to_board(self):
        # now that we have the bombs planted, let's assign a number 0-8 for all the empty spaces, which
        # represents how many neighboring bombs there are. we can precompute these and it'll save us some
        # effort checking what's around the board later on :)
        for r in range(self.dim_size):
            for c in range(self.dim_size):
                if self.board[r][c] == '*':
                    # if this is already a bomb, we don't want to calculate anything
                    continue
                self.board[r][c] = self.get_num_neighboring_bombs(r, c)

    def get_num_neighboring_bombs(self, row, col):
        # let's iterate through each of the neighboring positions and sum number of bombs
        # top left: (row-1, col-1)
        # top middle: (row-1, col)
        # top right: (row-1, col+1)
        # left: (row, col-1)
        # right: (row, col+1)
        # bottom left: (row+1, col-1)
        # bottom middle: (row+1, col)
        # bottom right: (row+1, col+1)

        # make sure to not go out of bounds!

        num_neighboring_bombs = 0
        for r in range(max(0, row-1), min(self.dim_size-1, row+1)+1):
            for c in range(max(0, col-1), min(self.dim_size-1, col+1)+1):
                if r == row and c == col:
                    # our original location, don't check
                    continue
                if self.board[r][c] == '*':
                    num_neighboring_bombs += 1

        return num_neighboring_bombs

    def dig(self, row, col):
        # dig at that location!
        # return True if successful dig, False if bomb dug

        # a few scenarios:
        # hit a bomb -> game over
        # dig at location with neighboring bombs -> finish dig
        # dig at location with no neighboring bombs -> recursively dig neighbors!

        self.dug.add((row, col)) # keep track that we dug here

        if self.board[row][col] == '*':
            return False
        elif self.board[row][col] > 0:
            return True

        # self.board[row][col] == 0
        for r in range(max(0, row-1), min(self.dim_size-1, row+1)+1):
            for c in range(max(0, col-1), min(self.dim_size-1, col+1)+1):
                if (r, c) in self.dug:
                    continue # don't dig where you've already dug
                self.dig(r, c)

        # if our initial dig didn't hit a bomb, we *shouldn't* hit a bomb here
        return True

    def __str__(self):
        # this is a magic function where if you call print on this object,
        # it'll print out what this function returns!
        # return a string that shows the board to the player

        # first let's create a new array that represents what the user would see
        visible_board = [[None for _ in range(self.dim_size)] for _ in range(self.dim_size)]
        for row in range(self.dim_size):
            for col in range(self.dim_size):
                if (row,col) in self.dug:
                    visible_board[row][col] = str(self.board[row][col])
                else:
                    visible_board[row][col] = ' '
        
        # put this together in a string
        string_rep = ''
        # get max column widths for printing
        widths = []
        for idx in range(self.dim_size):
            columns = map(lambda x: x[idx], visible_board)
            widths.append(
                len(
                    max(columns, key = len)
                )
            )

        # print the csv strings
        indices = [i for i in range(self.dim_size)]
        indices_row = '   '
        cells = []
        for idx, col in enumerate(indices):
            format = '%-' + str(widths[idx]) + "s"
            cells.append(format % (col))
        indices_row += '  '.join(cells)
        indices_row += '  \n'
        
        for i in range(len(visible_board)):
            row = visible_board[i]
            string_rep += f'{i} |'
            cells = []
            for idx, col in enumerate(row):
                format = '%-' + str(widths[idx]) + "s"
                cells.append(format % (col))
            string_rep += ' |'.join(cells)
            string_rep += ' |\n'

        str_len = int(len(string_rep) / self.dim_size)
        string_rep = indices_row + '-'*str_len + '\n' + string_rep + '-'*str_len

        return string_rep

# play the game
def play(dim_size=10, num_bombs=10):
    # Step 1: create the board and plant the bombs
    board = Board(dim_size, num_bombs)

    # Step 2: show the user the board and ask for where they want to dig
    # Step 3a: if location is a bomb, show game over message
    # Step 3b: if location is not a bomb, dig recursively until each square is at least
    #          next to a bomb
    # Step 4: repeat steps 2 and 3a/b until there are no more places to dig -> VICTORY!
    safe = True 

    while len(board.dug) < board.dim_size ** 2 - num_bombs:
        print(board)
        # 0,0 or 0, 0 or 0,    0
        user_input = re.split(',(\\s)*', input("Where would you like to dig? Input as row,col: "))  # '0, 3'
        row, col = int(user_input[0]), int(user_input[-1])
        if row < 0 or row >= board.dim_size or col < 0 or col >= dim_size:
            print("Invalid location. Try again.")
            continue

        # if it's valid, we dig
        safe = board.dig(row, col)
        if not safe:
            # dug a bomb ahhhhhhh
            break # (game over rip)

    # 2 ways to end loop, lets check which one
    if safe:
        print("CONGRATULATIONS!!!! YOU ARE VICTORIOUS!")
    else:
        print("SORRY GAME OVER :(")
        # let's reveal the whole board!
        board.dug = [(r,c) for r in range(board.dim_size) for c in range(board.dim_size)]
        print(board)

if __name__ == '__main__': # good practice :)
    play()

#9)SUDOKU 

from pprint import pprint


def find_next_empty(puzzle):
    # finds the next row, col on the puzzle that's not filled yet --> rep with -1
    # return row, col tuple (or (None, None) if there is none)

    # keep in mind that we are using 0-8 for our indices
    for r in range(9):
        for c in range(9): # range(9) is 0, 1, 2, ... 8
            if puzzle[r][c] == -1:
                return r, c

    return None, None  # if no spaces in the puzzle are empty (-1)

def is_valid(puzzle, guess, row, col):
    # figures out whether the guess at the row/col of the puzzle is a valid guess
    # returns True or False

    # for a guess to be valid, then we need to follow the sudoku rules
    # that number must not be repeated in the row, column, or 3x3 square that it appears in

    # let's start with the row
    row_vals = puzzle[row]
    if guess in row_vals:
        return False # if we've repeated, then our guess is not valid!

    # now the column
    # col_vals = []
    # for i in range(9):
    #     col_vals.append(puzzle[i][col])
    col_vals = [puzzle[i][col] for i in range(9)]
    if guess in col_vals:
        return False

    # and then the square
    row_start = (row // 3) * 3 # 10 // 3 = 3, 5 // 3 = 1, 1 // 3 = 0
    col_start = (col // 3) * 3

    for r in range(row_start, row_start + 3):
        for c in range(col_start, col_start + 3):
            if puzzle[r][c] == guess:
                return False

    return True

def solve_sudoku(puzzle):
    # solve sudoku using backtracking!
    # our puzzle is a list of lists, where each inner list is a row in our sudoku puzzle
    # return whether a solution exists
    # mutates puzzle to be the solution (if solution exists)
    
    # step 1: choose somewhere on the puzzle to make a guess
    row, col = find_next_empty(puzzle)

    # step 1.1: if there's nowhere left, then we're done because we only allowed valid inputs
    if row is None:  # this is true if our find_next_empty function returns None, None
        return True 
    
    # step 2: if there is a place to put a number, then make a guess between 1 and 9
    for guess in range(1, 10): # range(1, 10) is 1, 2, 3, ... 9
        # step 3: check if this is a valid guess
        if is_valid(puzzle, guess, row, col):
            # step 3.1: if this is a valid guess, then place it at that spot on the puzzle
            puzzle[row][col] = guess
            # step 4: then we recursively call our solver!
            if solve_sudoku(puzzle):
                return True
        
        # step 5: it not valid or if nothing gets returned true, then we need to backtrack and try a new number
        puzzle[row][col] = -1

    # step 6: if none of the numbers that we try work, then this puzzle is UNSOLVABLE!!
    return False

if __name__ == '__main__':
    example_board = [
        [3, 9, -1,   -1, 5, -1,   -1, -1, -1],
        [-1, -1, -1,   2, -1, -1,   -1, -1, 5],
        [-1, -1, -1,   7, 1, 9,   -1, 8, -1],

        [-1, 5, -1,   -1, 6, 8,   -1, -1, -1],
        [2, -1, 6,   -1, -1, 3,   -1, -1, -1],
        [-1, -1, -1,   -1, -1, -1,   -1, -1, 4],

        [5, -1, -1,   -1, -1, -1,   -1, -1, -1],
        [6, 7, -1,   1, -1, 5,   -1, 4, -1],
        [1, -1, 9,   -1, -1, -1,   2, -1, -1]
    ]
    print(solve_sudoku(example_board))
    pprint(example_board)


#10) PHOTOEDITING


from image import Image
import numpy as np

def brighten(image, factor):
    # when we brighten, we just want to make each channel higher by some amount 
    # factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!

    # # this is the non vectorized version
    # for x in range(x_pixels):
    #     for y in range(y_pixels):
    #         for c in range(num_channels):
    #             new_im.array[x, y, c] = image.array[x, y, c] * factor

    # faster version that leverages numpy
    new_im.array = image.array * factor

    return new_im

def adjust_contrast(image, factor, mid):
    # adjust the contrast by increasing the difference from the user-defined midpoint by factor amount
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_im.array[x, y, c] = (image.array[x, y, c] - mid) * factor + mid

    return new_im

def blur(image, kernel_size):
    # kernel size is the number of pixels to take into account when applying the blur
    # (ie kernel_size = 3 would be neighbors to the left/right, top/bottom, and diagonals)
    # kernel size should always be an *odd* number
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    neighbor_range = kernel_size // 2  # this is a variable that tells us how many neighbors we actually look at (ie for a kernel of 3, this value should be 1)
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                # we are going to use a naive implementation of iterating through each neighbor and summing
                # there are faster implementations where you can use memoization, but this is the most straightforward for a beginner to understand
                total = 0
                for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size ** 2)
    return new_im

def apply_kernel(image, kernel):
    # the kernel should be a 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    neighbor_range = kernel.shape[0] // 2  # this is a variable that tells us how many neighbors we actually look at (ie for a 3x3 kernel, this value should be 1)
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_im.array[x, y, c] = total
    return new_im

def combine_images(image1, image2):
    # let's combine two images using the squared sum of squares: value = sqrt(value_1**2, value_2**2)
    # size of image1 and image2 MUST be the same
    x_pixels, y_pixels, num_channels = image1.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_im.array[x, y, c] = (image1.array[x, y, c]**2 + image2.array[x, y, c]**2)**0.5
    return new_im
    
if __name__ == '__main__':
    lake = Image(filename='lake.png')
    city = Image(filename='city.png')

    # brightening
    brightened_im = brighten(lake, 1.7)
    brightened_im.write_image('brightened.png')

    # darkening
    darkened_im = brighten(lake, 0.3)
    darkened_im.write_image('darkened.png')

    # increase contrast
    incr_contrast = adjust_contrast(lake, 2, 0.5)
    incr_contrast.write_image('increased_contrast.png')

    # decrease contrast
    decr_contrast = adjust_contrast(lake, 0.5, 0.5)
    decr_contrast.write_image('decreased_contrast.png')

    # blur using kernel 3
    blur_3 = blur(city, 3)
    blur_3.write_image('blur_k3.png')

    # blur using kernel size of 15
    blur_15 = blur(city, 15)
    blur_15.write_image('blur_k15.png')

    # let's apply a sobel edge detection kernel on the x and y axis
    sobel_x = apply_kernel(city, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    sobel_x.write_image('edge_x.png')
    sobel_y = apply_kernel(city, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_y.write_image('edge_y.png')

    # let's combine these and make an edge detector!
    sobel_xy = combine_images(sobel_x, sobel_y)
    sobel_xy.write_image('edge_xy.png')


#11) BINARY SEARCH



import random
import time

# Implementation of binary search algorithm!!

# We will prove that binary search is faster than naive search!


# Essence of binary search:
# If you have a sorted list and you want to search this array for something,
# You could go through each item in the list and ask, is this equal to what we're looking for?
# But we can make this *faster* by leveraging the fact that our array is sorted!
# Binary search ~ O(log(n)), naive search ~ O(n)

# In these two examples, l is a list in ascending order, and target is something that we're looking for
# Return -1 if not found


# naive search: scan entire list and ask if its equal to the target
# if yes, return the index
# if no, then return -1
def naive_search(l, target):
    # example l = [1, 3, 10, 12]
    for i in range(len(l)):
        if l[i] == target:
            return i
    return -1


# binary search uses divide and conquer!
# we will leverage the fact that our list is SORTED
def binary_search(l, target, low=None, high=None):
    if low is None:
        low = 0
    if high is None:
        high = len(l) - 1

    if high < low:
        return -1

    # example l = [1, 3, 5, 10, 12]  # should return 3
    midpoint = (low + high) // 2  # 2

    # we'll check if l[midpoint] == target, and if not, we can find out if
    # target will be to the left or right of midpoint
    # we know everything to the left of midpoint is smaller than the midpoint
    # and everything to the right is larger
    if l[midpoint] == target:
        return midpoint
    elif target < l[midpoint]:
        return binary_search(l, target, low, midpoint-1)
    else:
        # target > l[midpoint]
        return binary_search(l, target, midpoint+1, high)

if __name__=='__main__':
    # l = [1, 3, 5, 10, 12]
    # target = 7
    # print(naive_search(l, target))
    # print(binary_search(l, target))

    length = 10000
    # build a sorted list of length 10000
    sorted_list = set()
    while len(sorted_list) < length:
        sorted_list.add(random.randint(-3*length, 3*length))
    sorted_list = sorted(list(sorted_list))

    start = time.time()
    for target in sorted_list:
        naive_search(sorted_list, target)
    end = time.time()
    print("Naive search time: ", (end - start), "seconds")

    start = time.time()
    for target in sorted_list:
        binary_search(sorted_list, target)
    end = time.time()
    print("Binary search time: ", (end - start), "seconds")

 
#12) MARKOV CHAIN COMPOSER 


import os
import re
import string
import random
from graph import Graph, Vertex

def get_words_from_text(text_path):
    with open(text_path, 'rb') as file:
        text = file.read().decode("utf-8") 

        # remove [verse 1: artist]
        # include the following line if you are doing song lyrics
        # text = re.sub(r'\[(.+)\]', ' ', text)

        text = ' '.join(text.split())
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()

    words = words[:1000]

    return words


def make_graph(words):
    g = Graph()
    prev_word = None
    # for each word
    for word in words:
        # check that word is in graph, and if not then add it
        word_vertex = g.get_vertex(word)

        # if there was a previous word, then add an edge if does not exist
        # if exists, increment weight by 1
        if prev_word:  # prev word should be a Vertex
            # check if edge exists from previous word to current word
            prev_word.increment_edge(word_vertex)

        prev_word = word_vertex

    g.generate_probability_mappings()
    
    return g

def compose(g, words, length=50):
    composition = []
    word = g.get_vertex(random.choice(words))
    for _ in range(length):
        composition.append(word.value)
        word = g.get_next_word(word)

    return composition


def main():
    words = get_words_from_text('texts/hp_sorcerer_stone.txt')

    # for song in os.listdir('songs/{}'.format(artist)):
        # if song == '.DS_Store':
        #     continue
        # words.extend(get_words_from_text('songs/{artist}/{song}'.format(artist=artist, song=song)))
        
    g = make_graph(words)
    composition = compose(g, words, 100)
    print(' '.join(composition))


if __name__ == '__main__':
    main()

#----END----    