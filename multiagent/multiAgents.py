# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math


from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        "*** MY CODE HERE ***"
        newScore = successorGameState.getScore()*5 #as we are score oriented, I multiply it by 5 to increase it's weight

        # Food statistics
        newFoodLocations = newFood.asList()
        currNumberOfFoods = len(currentGameState.getFood().asList())
        newNumberOfFoods = len(newFoodLocations)

        # Capsule statistics
        newCapsulesLocations = successorGameState.getCapsules()
        currNumberOfCapsules = len(currentGameState.getCapsules())
        newNumberOfCapsules = len(newCapsulesLocations)

        if currNumberOfCapsules > newNumberOfCapsules or not newCapsulesLocations:
            nearestCapsuleDistance = -100 # When it gets subtracted it will have a big enough effect to make us eat the capsule right next to us
        else:
            # as we want to get closer to capsule, we need to keep track of the distance we have to the nearest food
            if newCapsulesLocations:
                nearestCapsuleDistance = min([manhattanDistance(capsule, newPos) for capsule in newCapsulesLocations])
            else:
                nearestCapsuleDistance = 0

        if currNumberOfFoods > newNumberOfFoods or not newFoodLocations:
            nearestFoodsDistance = -50 # When it gets subtracted it will have a big enough effect to make us eat the food right next to us
        else:
            # as we want to get closer to food, we need to keep track of the distance we have to the nearest food
            if newFoodLocations:
                nearestFoodsDistance = min([manhattanDistance(food, newPos) for food in newFoodLocations])/2
            #the reason I divide it by 2 is two foods on the other side of the wall can make my pacman get stuck, so I don't want
            #them having the same distance have a major affect on the final decision
            else:
                nearestFoodsDistance = 0

        if newGhostStates: #if ghost present
            # We must be careful when we are taking ghosts in to consider, there are two types we want to deal:
            edibleGhostDistance = None #shortest distance to the one we can digest
            nonedibleGhostDistance = None #shortest distance to the one we need to run away from
            newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
            if sum(newScaredTimes) == 0: #this is inorder to be fast, no ghost to eat means all of them belong to nonedible
                nonedibleGhostDistance = min([manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates])
                if nonedibleGhostDistance == 1:  # Under any circumstance we want to avoid running into a ghost
                    return -999999999
            else:
                currNumberOfGhosts = len(currentGameState.getGhostStates())
                newNumberOfGhosts = len(newGhostStates)
                for ghost in newGhostStates:
                    ghostDistance = manhattanDistance(ghost.getPosition(), newPos) #distance of a ghost
                    if ghost.scaredTimer <= 1:
                        if ghostDistance == 1: #if we have a nonedible ghost collusion we avoid this successor state
                            return -999999999
                        if nonedibleGhostDistance == None or nonedibleGhostDistance > ghostDistance:
                            nonedibleGhostDistance = ghostDistance
                    else:
                        if currNumberOfGhosts > newNumberOfGhosts:
                            edibleGhostDistance = -50
                        elif edibleGhostDistance == None or edibleGhostDistance > ghostDistance:
                            edibleGhostDistance = ghostDistance

        if edibleGhostDistance != None:
            # we want to minimize the distance we have to the edible ghost, just like food but I just give more weight to them
            newScore -= edibleGhostDistance
        if nonedibleGhostDistance != None:
            #we want to runaway from the ghost, so that's why it's a plus
            newScore += math.log1p(1+nonedibleGhostDistance)
            #we thake the ln(1+distance) because when their distance is far, the affect of their movement becomes less and when they are near, it's importance raises exponentially

        # Sometimes the weight between the distances of food and ghosts can balance the 2*1 point we loose by stopping, and I don't prefer that to happen
        if action == 'Stop':
            if newScore > 0:
                newScore /=2
            else:
                newScore *=2

        return newScore - (nearestFoodsDistance + nearestCapsuleDistance)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** MY CODE HERE ***"
        currState = gameState
        possiblePacmanMoves = currState.getLegalActions(0)
        possiblePacmanStates = [currState.generateSuccessor(0, action) for action in possiblePacmanMoves]
        #after we create all the possible states for the possible pacman moves, we want to choose the one with the max score
        possiblePacmanScores =[self.collectTreeScores(state, 1, 0) for state in possiblePacmanStates]
        #after we get the score of each state, we find the maximum one and it's index, then return the action corresponding to that subtree
        bestScore = max(possiblePacmanScores)
        bestIndices = [index for index in range(len(possiblePacmanScores)) if possiblePacmanScores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return possiblePacmanMoves[chosenIndex]

    def collectTreeScores(self, gameState,agentIndex, depth):

        if agentIndex == gameState.getNumAgents():
            depth +=1 #well if we got back to the first agent, we dived one level deeper

        agentIndex = agentIndex % gameState.getNumAgents() #a quick fix to agentIndex

        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose(): #these mean we've reached the end
            return self.evaluationFunction(gameState)

        nextStates = [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        nextStateScores = [self.collectTreeScores(state, agentIndex + 1, depth) for state in nextStates]

        if agentIndex > 0:
            """ MIN """
            return min(nextStateScores)
        else:
            """ MAX """
            return max(nextStateScores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** MY CODE HERE ***"
        currState = gameState
        possiblePacmanMoves = currState.getLegalActions(0)
        possiblePacmanScores = []
        # after we create all the possible states for the possible pacman moves, we want to choose the one with the max score
        alpha = -99999999999
        beta = 99999999999
        """ MAX """
        for action in possiblePacmanMoves:
            temp = self.collectTreeScores(currState.generateSuccessor(0,action), 1, 0, alpha, beta)
            possiblePacmanScores.append(temp)
            alpha = max(alpha,temp)

        # after we get the score of each state, we find the maximum one and it's index, then return the action corresponding to that subtree
        bestScore = max(possiblePacmanScores)
        bestIndices = [index for index in range(len(possiblePacmanScores)) if possiblePacmanScores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return possiblePacmanMoves[chosenIndex]


    def collectTreeScores(self, gameState, agentIndex, depth, alpha, beta):

        if agentIndex == gameState.getNumAgents():
            depth += 1  # well if we got back to the first agent, we dived one level deeper

        agentIndex = agentIndex % gameState.getNumAgents()  # a quick fix to agentIndex

        legalActions = gameState.getLegalActions(agentIndex)

        if depth == self.depth or not legalActions or gameState.isWin() or gameState.isLose():  # these mean we've reached the end
            return self.evaluationFunction(gameState)

        if agentIndex > 0: #GHOSTS
            """ MIN """
            temp = 99999999999
            for action in legalActions:
                temp = min(temp, self.collectTreeScores(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                if temp < alpha:
                    return temp
                beta = min(beta, temp)
            return temp

        else: #PACMAN
            """ MAX """
            temp = -99999999999
            for action in legalActions:
                temp = max(temp, self.collectTreeScores(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                if temp > beta:
                    return temp
                alpha = max(alpha,temp)
            return temp


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** MY CODE HERE ***"
        currState = gameState
        possiblePacmanMoves = currState.getLegalActions(0)
        possiblePacmanStates = [currState.generateSuccessor(0, action) for action in possiblePacmanMoves]
        # after we create all the possible states for the possible pacman moves, we want to choose the one with the max score
        possiblePacmanScores = [self.collectTreeScores(state, 1, 0) for state in possiblePacmanStates]
        # after we get the score of each state, we find the maximum one and it's index, then return the action corresponding to that subtree
        bestScore = max(possiblePacmanScores)
        bestIndices = [index for index in range(len(possiblePacmanScores)) if possiblePacmanScores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return possiblePacmanMoves[chosenIndex]

    def collectTreeScores(self, gameState, agentIndex, depth):

        if agentIndex == gameState.getNumAgents():
            depth += 1  # well if we got back to the first agent, we dived one level deeper

        agentIndex = agentIndex % gameState.getNumAgents()  # a quick fix to agentIndex

        if depth == self.depth or not gameState.getLegalActions(
                agentIndex) or gameState.isWin() or gameState.isLose():  # these mean we've reached the end
            return self.evaluationFunction(gameState)

        nextStates = [gameState.generateSuccessor(agentIndex, action) for action in
                      gameState.getLegalActions(agentIndex)]
        nextStateScores = [self.collectTreeScores(state, agentIndex + 1, depth) for state in nextStates]

        if agentIndex > 0: # GHOSTS
            #taking average for the behaviour will be enough becayse it is said that there is uniform distribution
            return sum(nextStateScores) / float(len(nextStateScores))

        else: # PACMAN
            return max(nextStateScores)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <I have worked on these parameters:
      Score:
        -Need to maximize it and since it's our main aim, I multiply it with 5 to increase it's weight
      Food Distance:
        -We want to minimize it. In case there are two foods right after the wall, since Pacman cannot bypass
        the wall, it can get stuck in between states and in order to cancel these situations I lowered the weight
        of food distance, making other parameters more important compared to that situation and not letting it get
        stuck it between states
      Capsule Distance:
        -Same like food distance, need to minimize it but the capsules are rarer than food so they don't cause the same
        problem, that's why I didn't rearragne it's weight
      Edible Ghost Distance:
        -Just like capsules but they are more delicious and tastyyyy
      Non Edible Ghost Distance:
        -They are the worst nightmares. I want my Pacman to be away from them. After calculating the difference of the
        closest one as x, I put it in ln(1+x). The reason I use ln is the distance we have to these ghosts are important
        when they are nearby. It's value decreases as they are further away. For instance, a change from distance 1 to 2
        is much more important than the change from 20 to 21.
      >
    """
    "*** MY CODE HERE ***"

    currScore = currentGameState.getScore() * 5

    if currentGameState.isWin(): #To be fast, when it's a win, we take it
        return 999999999999999

    if currentGameState.isLose():# And to be fast,  when it's a lose, we don't want to it
        #but if we have no options other than lose, we take the one with higher score
        if currScore > 0:
            return -999999999999999+currScore
        else:
            return -999999999999999-currScore

    currPos = currentGameState.getPacmanPosition()
    currFoods = currentGameState.getFood()


    # calculating pacman's distances to foods and capsules
    currFoodLocations = currFoods.asList()
    currCapsulesLocations = currentGameState.getCapsules()
    if currCapsulesLocations:
        nearestCapsuleDistance = min([manhattanDistance(capsule, currPos) for capsule in currCapsulesLocations])
    else:
        nearestCapsuleDistance = 0
    nearestFoodsDistance = min([manhattanDistance(food, currPos) for food in currFoodLocations]) / 2

    #Calculating the ghost distances, depending on the type of ghost
    edibleGhostDistance = None
    nonedibleGhostDistance = None
    currGhostStates = currentGameState.getGhostStates()

    if currGhostStates: #if there is at least one ghost present
        currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
        if sum(currScaredTimes) == 0: # this is just to be fast, if there is no sacred time, there isn't any edible ghost
            nonedibleGhostDistance = min([manhattanDistance(ghost.getPosition(), currPos) for ghost in currGhostStates])
        else:
            for ghost in currGhostStates:
                ghostDistance = manhattanDistance(ghost.getPosition(), currPos)  # distance of a ghost
                if ghost.scaredTimer <= 1: #non-edible ghost
                    if nonedibleGhostDistance == None or nonedibleGhostDistance > ghostDistance:
                        nonedibleGhostDistance = ghostDistance
                else: # delicious and tasty edible ghost
                    if edibleGhostDistance == None or edibleGhostDistance > ghostDistance:
                        edibleGhostDistance = ghostDistance

    if edibleGhostDistance != None:
        currScore -= edibleGhostDistance
    if nonedibleGhostDistance != None:
        # this is the deal-breaker: importance of ghost distance decreases as the ghost is further away
        currScore += math.log1p(1 + nonedibleGhostDistance)

    return currScore - (nearestFoodsDistance + nearestCapsuleDistance)

# Abbreviation
better = betterEvaluationFunction

