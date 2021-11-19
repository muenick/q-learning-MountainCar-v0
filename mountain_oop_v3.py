import numpy as np # 
import gym # Umgebung mit unterschiedlichen games für die man Agenten definieren muss
from gym import wrappers # 
from datetime import date # just to have timestamps in the files

class MountainCarRL: 

    ## Konstruktor
    def __init__(self, episodes=25000, discount=0.95, learning_rate=0.1, show_every=3000, drawVideo=True):
        self.LEARNING_RATE = learning_rate
        self.DISCOUNT = discount
        self.EPISODES = episodes
        self.SHOW_EVERY = show_every
        self.init_enviroment(drawVideo)
        self.init_qtable()
        self.init_exploration()

    ## erzeugt das Spiel Enviroment in dem das Spiel und verhaltensweise Implementiert sind, sowie die möglichen Aktionen und "Spielregeln"
    def init_enviroment(self,drawVideo):
        self.env = gym.make("MountainCar-v0")
        if drawVideo == True:
            self.draw_video()
    
    ## erzeugt die Videos und Statistik. Nutz die Funktionen der Wrapper vom gym.openai.com Package
    def draw_video(self):
        currentDate = date.today()
        self.env = wrappers.RecordEpisodeStatistics(self.env)
        self.env = wrappers.RecordVideo(self.env, './videos/'+ str(currentDate.year) + str(currentDate.month) + str(currentDate.day) + '/')

    ## erzeugt die qTable 
    def init_qtable(self):
        self.DISCRETE_OS_SIZE = [20] * len(self.env.observation_space.high)
        self.discrete_os_win_size = (self.env.observation_space.high - self.env.observation_space.low)/self.DISCRETE_OS_SIZE
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.DISCRETE_OS_SIZE + [self.env.action_space.n]))

        print(self.DISCRETE_OS_SIZE)
        print(self.discrete_os_win_size)
        print(self.q_table.shape)

    ## definition wie schnell die Quote von zufälligen Aktionen zu Aktionen die auf der qTable basieren sinken soll
    def init_exploration(self):
        # Exploration settings
        self.epsilon = 1  # not a constant, qoing to be decayed
        self.START_EPSILON_DECAYING = 1
        self.END_EPSILON_DECAYING = self.EPISODES//2
        self.epsilon_decay_value = self.epsilon/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

    ## gibt den dirscrenten State in dem intervall [[0: 20],[0: 20]] szurück
    def get_discrete_state(self,state):
        discrete_state = (state - self.env.observation_space.low)/self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

    ## Anzahl der Episoden die verwendet werden um die Q-Table anzulernen
    def start_learning(self):
        ## start einer neuen Episode
        for episode in range(self.EPISODES):
            discrete_state = self.get_discrete_state(self.env.reset())
            done = False
            if episode % self.SHOW_EVERY == 0:
                render = True
                print(episode)
            else:
                render = False
            while not done:
                ## durch zufällige Aktionen kann der Algorithmus anfangs schneller lernen
                ## da die qTable zu begin noch wenig aussagekräftig ist
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.q_table[discrete_state])
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.action_space.n)

                ## der State ist ein Tupel der Form (7, 10) 
                new_state, reward, done, _ = self.env.step(action)
                ## der State ist dann ein Tupel der Form: (7, 10)
                new_discrete_state = self.get_discrete_state(new_state)
                
                if episode % self.SHOW_EVERY == 0:
                    self.env.render()
                # If simulation did not end yet after last step - update Q table
                if not done:
                    # Maximum possible Q value in next step (for new state)
                    max_future_q = np.max(self.q_table[new_discrete_state])
                    # Current Q value (for current state and performed action)
                    current_q = self.q_table[discrete_state + (action,)]
                    # And here's our equation for a new Q value for current state and action
                    new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
                    # Update Q table with new Q value
                    self.q_table[discrete_state + (action,)] = new_q
                    print(new_discrete_state + (action,))
                # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
                elif new_state[0] >= self.env.goal_position:
                    #q_table[discrete_state + (action,)] = reward
                    self.q_table[discrete_state + (action,)] = 0
                discrete_state = new_discrete_state
            # Decaying is being done every episode if episode number is within decaying range
            if self.END_EPSILON_DECAYING >= episode >= self.START_EPSILON_DECAYING:
                self.epsilon -= self.epsilon_decay_value

    ## Abspeichern der qTable in einer Textdatei im selben Verzeichniss
    def print_qTable(self,filename):
        outFile = open(filename +".txt", "w")
        for i,velo in enumerate(self.q_table):
            for j, pos in enumerate(velo):
                outFile.write("State: (" + str(i)+ ", " + str(j) + ")")                
                outFile.write(str(pos) +"\n" )
        outFile.close()

    ## Schließen des Enviroments
    def close_enviroment(self):
        self.env.close()


### Ausführung des qLearning Algorithmus für MountainCar
mountainCar = MountainCarRL(10)

mountainCar.print_qTable("initQTable")
mountainCar.start_learning()
mountainCar.print_qTable("finishedQTable")