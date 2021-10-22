import operator
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

# Oggetto da minimizzare, eredita caratteristiche dalla classe "Fitness"
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Particella [lista] con 5 attributi:
#   1: fitness
#   2: velocità [lista]("speed")
#   3-4: smin e smax velocità min e max [lista]
#   5: miglior valore trovato finora (best)
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
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
def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))                    # same as: v_u1 = u1*(localbest - current position)
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))                         # same as: v_u2 = u2*(global best - current position)
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))     # same as: velocità = velocità + v_u1 + v_u2
    for i, speed in enumerate(part.speed):                                              
        if abs(speed) < part.smin:                                                      # aggiusta nel caso si vada < smin o >smax
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))                                 # ricalcola la posizione della particella sommandola con la velocità

# Creo il toolbox
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-100, pmax=100, smin=-5, smax=5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", benchmarks.bohachevsky)

# main:
def main():
    pop = toolbox.population(n=100)                                       # creo una popolazione di 100 particelle
    stats = tools.Statistics(lambda ind: ind.fitness.values)            # gestisco le statistiche
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()                                           # creo un Logbook per tenere conto dei risultati
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)                # calcolo la fitness per ogni particella alla sua attuale posizione
            if not part.best or part.best.fitness < part.fitness:       # aggiorno il local best se la fitness è migliore dei quella del precedente 
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:                 # aggiorno il global best se la fitness è migliore dei quella del precedente 
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)                                  # aggiorno tutte le particelle con le nuove posizioni

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return pop, logbook, best
    
    

if __name__ == "__main__":
    pop, logbook, best = main()
    print("Miglior Particella: ", best)
    print("fitness:", best.fitness.values)