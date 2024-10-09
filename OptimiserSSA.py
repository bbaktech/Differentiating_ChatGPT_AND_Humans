import random

import numpy as np
import keras
from keras.src.models import Sequential  
from keras.src.layers import Dense,Dropout,Flatten

BIG_SCORE = 1.e6  # type: float
NO_SPARROWS = 20
NO_PRD = NO_SPARROWS /3
SC = NO_SPARROWS *20 /100
NO_ITERATIONS = 5

class ProgressBar:
    #steps is number of iterations
    def __init__(self, steps, updates=100):
        self.step = 0.
        self.step_size = (updates / steps)
        self.updates = updates
        bar = self._make_bar(0)
        print(bar, end=' ')

    def update(self, i):
        self.step = i * self.step_size 
        bar = self._make_bar(i)
        print(bar, end=' ')
#        print(i)

    def done(self):
        self.step = self.updates
        bar = self._make_bar(self.updates)
        print(bar)

    def _make_bar(self, x):
        bar = "["
        for x in range(self.updates):
            print("\r", end=' ')
            bar += "=" if x < int(self.step) else " "
        bar += "]"
        return bar

class Sparrow:
    def __init__(self,x,y,structure, max_tokens):
        self.x = x
        self.y = y
        self.structure = structure
        self.max_tokens = max_tokens
        self.model = keras.models.model_from_json(self.structure)
        callbacks = [keras.callbacks.ModelCheckpoint("NLP_CNN01.keras",save_best_only=True)]
        self.model.fit(x,y, epochs=2, verbose=0, callbacks=callbacks)

    def isInDanger(self):
        return False

    def RandomFly(self):
        self.model = keras.models.model_from_json(self.structure)
        callbacks = [keras.callbacks.ModelCheckpoint("NLP_CNN01.keras",save_best_only=True)]
        self.model.fit(self.x,self.y, epochs=2, verbose=0, callbacks=callbacks)

    def SearchContinue(self):
        old_weights = self.model.get_weights()
        new_weights = [None] * len(old_weights)
        rn =  random.randint(0, 9)

        for i in range(len(old_weights)):
            if rn>4 :
                new_weights[i] = old_weights[i] + old_weights[i] / 100
            else:
                new_weights[i] = old_weights[i] - old_weights[i] / 100
    
        self.model.set_weights(new_weights)

    def SearchTowerdsSparrow(self, sp, deepth = 1):
        old_weights = self.model.get_weights()
        towords_weights = sp.model.get_weights()

        new_weights = [None] * len(old_weights)
        for i in range(len(old_weights)):
            new_weights[i] = old_weights[i]

        for i in range (deepth):
            rn =  random.randint(0,len(old_weights)-1)
            new_weights[rn] = towords_weights[rn]
    
        self.model.set_weights(new_weights)
    
    def get_score(self, x, y, update=True):
        score = self.model.evaluate(x, y,batch_size=32, verbose=0)
        return score
    
    def get_weights(self):
        return self.model.get_weights()

class OptimizerSSA:
    def __init__(self,model,loss,optimiser, max_tokens):
        self.optimiser = optimiser
        self.max_tokens = max_tokens
        self.n_sparrows = NO_SPARROWS
        self.structure = model.to_json()
        self.sparrows = [None] * NO_SPARROWS
        self.loss = loss
        self.length = len(model.get_weights())
       
        self.global_best_weights = None
        self.global_best_score = BIG_SCORE

    def fit(self, x, y, steps=NO_ITERATIONS, batch_size=32):

        for i in range(self.n_sparrows):
            self.sparrows[i] = Sparrow(x,y,structure=self.structure,max_tokens =self.max_tokens )

        for s in self.sparrows:
            score = s.get_score(x, y)
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_weights = s.get_weights()

        print("SSA -- train: {:.4f} ".format(self.global_best_score))
        bar = ProgressBar(steps)
        for i in range(steps):
            bar.update(i)
            old_list_sparrows = sorted(self.sparrows, key=lambda s1: s1.get_score(x,y))
            new_list_sparrows = sorted(self.sparrows, key=lambda s1: s1.get_score(x,y))

            BestSPARROWS = old_list_sparrows[0]
            WorstSPARROWS = old_list_sparrows[self.n_sparrows-1]
            best_score = old_list_sparrows[0].get_score(x, y)

            if best_score < self.global_best_score :
                self.global_best_score = best_score
                self.global_best_weights = BestSPARROWS.get_weights()

#           print("SSA -- iteration best score {:0.4f}".format(self.global_best_score))
            #sparrow - producers
            for j in range ( int(NO_PRD)):
                s2 = new_list_sparrows[j]
                #there are two options: continue search(2/4 steps) or fly to new location
                if (s2.isInDanger()):
                    s2.RandomFly()
                else:
                    s2.SearchContinue()

            #sparrow - scroungers
            for  j in range(int(NO_PRD),NO_SPARROWS):
                s2 = new_list_sparrows[j]
                #j < NO_SPARROWS/2 move near to best else move to new locations
                if (j < NO_SPARROWS/2):
                    s2.SearchTowerdsSparrow(BestSPARROWS)
                else:
                    s2.RandomFly()
        
            #sparrow - scouter
            for j in range( int(SC)):
                sc =  random.randint(0, self.n_sparrows-1)
                #once selected it should not be selected again - we will add this
                s2 = new_list_sparrows[sc]
                score1 = s2.get_score(x,y)				
                if (score1 > self.global_best_score):
                    s2.SearchTowerdsSparrow(BestSPARROWS)
                else:
                    s2.SearchContinue()

			#Archives the current locations of sparrows
            for j in  range(NO_SPARROWS):
                if (new_list_sparrows[j].get_score(x,y) < old_list_sparrows[j].get_score(x,y)):
                    old_list_sparrows[j]  = new_list_sparrows[j]

            self.sparrows = old_list_sparrows

        bar.done()

    def get_best_model(self):
        best_model = keras.models.model_from_json(self.structure)
        best_model.compile(loss=self.loss,optimizer=self.optimiser)
        best_model.set_weights(self.global_best_weights)
        return best_model
