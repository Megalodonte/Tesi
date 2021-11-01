import operator
import random
import math
from deap import base
from deap import creator

# Oggetto da minimizzare, eredita caratteristiche dalla classe "Fitness"
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Particella-classe [lista] con 5 attributi:
#   1: fitness
#   2: velocità [lista]("speed")
#   3-4: smin e smax velocità min e max [lista]
#   5: miglior valore trovato finora (best)
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)

# Genera una perticella con posizione casuale tra pmin e pmax e velocità casuale tra smin e smax 
# Dopo averla generata le assegna i valori di smin e smax, che di base sono "None"
# "size" è la dimensionalità dello spazio delle soluzioni
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

# Calcola la velocità e la nuova posizione della particella
def updateParticle(part, best, phi1, phi2, w):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))                    # same as: v_u1 = u1*(localbest - current position)
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))                         # same as: v_u2 = u2*(global best - current position)
    for i in part.speed:
        i = i * w
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))     # same as: velocità = velocità + v_u1 + v_u2
    for i, speed in enumerate(part.speed):                                              
        if abs(speed) < part.smin:                                                      # aggiusta nel caso si vada < smin o >smax
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))                                 # ricalcola la posizione della particella sommandola con la velocità

   