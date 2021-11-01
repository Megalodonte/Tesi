from deap import base
from deap import creator
from deap import tools
import  numpy
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import functions
import PSO

simplefilter("ignore", category=ConvergenceWarning)

# parametri
dataset_name = "iris"
test_size = 0.3
phi1 = 1.4960
phi2 = 1.4960
neurons_in_hidden = 100
pmin = -2.0
pmax = 2.0
smin = -0.5
smax = 0.5
num_inputs = 4
num_outputs = 3
size = (num_inputs+num_outputs)*neurons_in_hidden + num_outputs + neurons_in_hidden
size_pop = 30
generations = 100
w = 0.7298
# preparazione
X_train, X_test, y_train, y_test = functions.load_dataset(dataset_name=dataset_name, test_size=test_size)

toolbox = base.Toolbox()
toolbox.register("particle", PSO.generate, size=size, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", PSO.updateParticle, phi1=phi1, phi2=phi2, w=w)
toolbox.register("evaluate", functions.test_weights_sklearn, X=X_train, y=y_train, neurons_in_hidden=neurons_in_hidden, inputs=num_inputs, outputs=num_outputs)

# main
def main():
    pop = toolbox.population(n=size_pop)                                # creo una popolazione di 100 particelle
    stats = tools.Statistics(lambda ind: ind.fitness.values)            # gestisco le statistiche
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()                                           # creo un Logbook per tenere conto dei risultati
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = generations
    best = None

    for g in range(GEN):
        
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)                        # calcolo la fitness per ogni particella alla sua attuale posizione
            if not part.best or part.best.fitness < part.fitness:               # aggiorno il local best se la fitness è migliore dei quella del precedente 
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:                         # aggiorno il global best se la fitness è migliore dei quella del precedente 
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)                                          # aggiorno tutte le particelle con le nuove posizioni

        # stampo le statistiche
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return pop, logbook, best

# avvio programma
if __name__ == "__main__":
    pop, logbook, best = main()
    print("fitness = ", best.fitness.values)
    print("Test finale:")
    fitness = functions.test_weights_sklearn(best, X=X_test, y=y_test, 
                neurons_in_hidden=neurons_in_hidden, inputs=num_inputs, outputs=num_outputs)
    print("fitness sul test set =", fitness)
