import math 
import random # we just imported math and random 

class Player: # we defined a class for player 
    def __init__(self,letter): #defined a fn which will have the x or o letter for game of tictactoe
        #letter is x or o 
        self.letter=letter
    def get_move(self,game): # we want all player to get next move given a game 
        pass

class RandomComputerPlayer(Player): #inheritance 
    def __init__(self, letter):    #the above class are only called 
        super().__init__(letter)
    def get_move(self, game):
        #get a random valid spot for our next move 
        square = random.choice(game.avaliable_moves())
        return square 

class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
    def get_move(self, game):
        valid_square= False
        val= None
        while not valid_square:
            square = input(self.letter+'\'s turn.Input move(0-9):')
            #we are goin to check that thhis is a correct value by trying to cast 
            #it to an integer and if it s not then we say its invalid 
            #if thta spot is not available on the board then also w esay its invalid 
            try :
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square= True # if these are succesful , then yay ! 
            except ValueError:
                print('Invalid square. Try again')

        return val        

