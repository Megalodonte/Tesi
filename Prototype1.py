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
dataset_name = "liver"
test_size = 0.3
phi1 = 2.1
phi2 = 2.1
wmax = 0.9
wmin = 0.6
pmin = -2
pmax = 2
smin = -1
smax = 1
size_pop = 50
generations = 200
num_tests = 15

# preparazione
X_train, X_test, y_train, y_test, num_inputs, num_outputs, neurons_in_hidden = functions.load_dataset(dataset_name=dataset_name, test_size=test_size)
size = functions.get_size(num_inputs, num_outputs, neurons_in_hidden)

# toolbox
toolbox = base.Toolbox()
toolbox.register("particle", PSO.generate, size=size, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", PSO.updateParticle, phi1=phi1, phi2=phi2)
toolbox.register("evaluate", functions.test_weights_sklearn, X=X_train, y=y_train, neurons_in_hidden=neurons_in_hidden, inputs=num_inputs, outputs=num_outputs, id=True)

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

        wtemp = wmax - ((wmax - wmin)*g)/GEN
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)                        # calcolo la fitness per ogni particella alla sua attuale posizione
            if not part.best or part.best.fitness < part.fitness:               # aggiorno il local best se la fitness è migliore dei quella del precedente
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:                         # aggiorno il global best se la fitness è migliore dei quella del precedente
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best, w=wtemp)                                          # aggiorno tutte le particelle con le nuove posizioni

        # stampo le statistiche
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    return pop, logbook, best

# avvio programma
if __name__ == "__main__":

    train_vector = []
    test_vector = []
    logbooks = []
    w_and_b = []
    for i in range(num_tests):
        print("--------Test %d/%d--------" %(i+1,num_tests))
        pop, logbook, best = main()
        logbooks.append(logbook)
        print("Fitness finale del migliore = ", best.fitness.values[0])
        train_vector.append(best.fitness.values[0])
        fitness = functions.test_weights_sklearn(best, X=X_test, y=y_test,
                    neurons_in_hidden=neurons_in_hidden, inputs=num_inputs, outputs=num_outputs, id=False)
        print("Accuracy sul test set =", fitness[0])
        test_vector.append(fitness[0])
        w_and_b.append(best)
    functions.save_test(dataset_name, train_vector, test_vector, w_and_b, logbooks)
    print("--------Fine dei test--------")