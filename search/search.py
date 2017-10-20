# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** MY CODE BEGINS HERE ***"

    currState = problem.getStartState() #We get our initial State as currState
    ourStack = util.Stack()  #We initiliaze a stack
    ourStack.push([[currState]]) #we push our start state to ourStack 
    visitedSet = [] #we'll do some book keeping so that we don't do cycles

    while (ourStack.isEmpty() != True): #We loop till we make sure our search is complete
        
        currList = ourStack.pop() #Get the top list on the stack
        currState = currList[len(currList)-1] #Pull the last state from that list, this is  what we will be working on
        
        if currState[0] not in visitedSet: #if these coordinates are not visited we visit them
            visitedSet.append(currState[0]) #we mark these new coordinats as visited
            curSuccessors = problem.getSuccessors(currState[0]) #For that currState, we get the neighbours
            unvisitedSuccessors = [] #here we will seperate the neighbours which are not visited yeter
            for i in range(len(curSuccessors)):
                if curSuccessors[i][0] not in visitedSet:
                    unvisitedSuccessors.append(curSuccessors[i])

            if (len(unvisitedSuccessors) > 0): #if we have unvisited neighbours, we check the last one so if it's the answer we can just 
            #figure it out here, because that will be the one we will pop from the stack next iteration anyway
                lastNeighbour = unvisitedSuccessors[len(unvisitedSuccessors)-1]
                
                if problem.isGoalState(lastNeighbour[0]): #if it's the goal state we found it
                        tempList = currList + [lastNeighbour]
                        answer = [] #we create our answer list which we will return 
                        for i in range(len(tempList)-1): #we just copy the actions because they tell pacman what to do
                            stateToGo = tempList[i+1]
                            answer.append(stateToGo[1])
                        return answer #finally we return the answer

            for neighbourState in unvisitedSuccessors: #if we couldn't find the goal, we proceed to each of the unvisited neighbours
                tempList = [] + currList #we create a copy of the current list
                tempList.append(neighbourState) #then we add the neighbour
                ourStack.push(tempList) #we push that tempList for further explorations

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** MY CODE BEGINS HERE ***"
    currState = problem.getStartState() #We get our initial State as currState again
    ourQueue = util.Queue()  #We initiliaze a queue this time
    ourQueue.push([(currState,None,0)]) #we push our start state to ourQueue 
    visitedSet = [] #keeping track of the coordinates which are already visited
    while (ourQueue.isEmpty() != True): #We loop till we make sure our search is complete
        currList = ourQueue.pop() #Get the front list in ourQueue
        currState = currList[len(currList)-1] #Pull the last state from that list, this is  what we will be working on
        if currState[0] not in visitedSet: #if it is not visited we pay a visit
            visitedSet.append(currState[0]) #first we mark it as visited 
            if problem.isGoalState(currState[0]): #then if it's the goal state we found it
                answer = [] #we create our answer list which we will return as an answer
                for i in range(len(currList)-1): #we just gather the actions as an answer
                    stateToGo = currList[i+1]
                    answer.append(stateToGo[1])
                return answer #finally we return the answer
            #if we couldn't find the goal state, we need to further explore
            curSuccessors = problem.getSuccessors(currState[0]) #For that currState, we get the neighbours
            unvisitedSuccessors = []
            for i in range(len(curSuccessors)): #we distinguish the unvisited neighbours for proceeding correctly
                if curSuccessors[i][0] not in visitedSet:
                    unvisitedSuccessors.append(curSuccessors[i])
            for neighbourState in unvisitedSuccessors: #for each of these neighbours          
                tempList = [] + currList #we create a copy of the current list
                tempList.append(neighbourState) #then we add the neighbour
                ourQueue.push(tempList) #we push that tempList for further explorations

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** MY CODE BEGINS HERE ***"
    currState = problem.getStartState() #our beginning state is called currState again
    ourPriorityQueue = util.PriorityQueue()  #We initiliaze a priority queue this time because the costs matter
    ourPriorityQueue.push([[currState,None,0]],0) #we push our start state to our PriorityQueue with 0 cost because it doesn't matter anyway 
    visitedSet = [] #we'll keep doing bookkeeping

    while (ourPriorityQueue.isEmpty() != True): #as long as our priority queue has a list, we have a chance of finding our goal
        currList = ourPriorityQueue.pop() #we get the first list in the queue, which has the least cost
        currState = currList[len(currList)-1] #we seperate it's last element to work with
    
        if currState[0] not in visitedSet: #if this element is not visited yet
            visitedSet.append(currState[0]) #we mark it as visited
            
            if problem.isGoalState(currState[0]): #and if we are lucky and if it's the goal state we found it
                answer = [] #we create our answer list which we will return as an answer
                for i in range(len(currList)-1):
                    stateToGo = currList[i+1]
                    answer.append(stateToGo[1])
                return answer #finally we return the answer

            curSuccessors = problem.getSuccessors(currState[0]) #to proceed further we call the neighbours
            unvisitedSuccessors = []
            for i in range(len(curSuccessors)): #we get the unvisited neighbours only thanks to our bookkeeping
                if curSuccessors[i][0] not in visitedSet:
                    unvisitedSuccessors.append(curSuccessors[i])
            #in order to push new lists to our priority queue, we'll  need to calculate the cost of this list
            curCost = 0 
            for i in range(len(currList)):
                curCost = curCost + currList[i][2]

            for neighbourState in unvisitedSuccessors: #for each of the unvisited neighbours
                    tempList = [] + currList #we create a copy of the current list
                    tempList.append(neighbourState) #then we add the neighbour
                    tempCost = curCost + neighbourState[2] #we add the current cost with the neighbours cost and then
                    ourPriorityQueue.push(tempList,tempCost) #we push that tempList with the calcualted cost for further explorations

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** MY CODE BEGINS HERE ***"
    currState = problem.getStartState() #our beginning state is called currState again
    ourPriorityQueue = util.PriorityQueue()  #We initiliaze a priority queue just like UCS
    ourPriorityQueue.push([(currState,None,0)],heuristic(currState,problem)) #we push our initial state to the PQueue with heuristic function, because we can
    visitedSet = [] #bookkeeping is still required

    while (ourPriorityQueue.isEmpty() != True):
        currList = ourPriorityQueue.pop()
        currState = currList[len(currList)-1]
        
        if currState[0] not in visitedSet:
            visitedSet.append(currState[0])
            
            if problem.isGoalState(currState[0]): #if it's the goal state we found it
                answer = [] #we create our answer list which we will return as an answer
                for i in range(len(currList)-1):
                    stateToGo = currList[i+1]
                    answer.append(stateToGo[1])
                return answer #finally we return the answer

            curSuccessors = problem.getSuccessors(currState[0])
            unvisitedSuccessors = []
            for i in range(len(curSuccessors)):
                if curSuccessors[i][0] not in visitedSet:
                    unvisitedSuccessors.append(curSuccessors[i])

            curCost = 0 #we calculate the currentcost before we push to our Pqueue
            for i in range(len(currList)):
                curCost = curCost + currList[i][2]

            for neighbourState in unvisitedSuccessors: #for each of these neighbours
                    tempList = [] + currList #we create a copy of the current list
                    tempList.append(neighbourState) #then we add the neighbour
                    #as we have the current cost and the neighbors cost now we add them with the heuristics
                    tempCost = curCost + neighbourState[2] + heuristic(neighbourState[0],problem)
                    ourPriorityQueue.push(tempList,tempCost) #we push that tempList for further explorations
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
