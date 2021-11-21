dataset_name = "fff"
if 1: print("e")
elif dataset_name == "digits":
        dataset = load_digits()
        inputs = 64
        outputs = 10

elif dataset_name == "glass":
        d, t = get_arrays("datasets/glass/glass.csv", "Type")
        dataset = ds(data=d, target=t)
        inputs = 9
        outputs = 7

elif dataset_name == "ph":
        d, t = get_arrays("datasets/ph/ph-data.csv", "label")
        dataset = ds(data=d, target=t)
        inputs = 3
        outputs = 15

elif dataset_name == "teaching":
        d, t = get_arrays("datasets/teaching/teaching.csv", "class", first_column=True)
        dataset = ds(data=d, target=t)
        inputs = 5
        outputs = 3

elif dataset_name == "car":
        d, t = get_arrays("datasets/car/car.csv", "class")
        dataset = ds(data=d, target=t)
        inputs = 6
        outputs = 4

elif dataset_name == "hayes":
        d, t = get_arrays("datasets/hayes/hayes.csv", "class")
        dataset = ds(data=d, target=t)
        inputs = 4
        outputs = 3

elif dataset_name == "monk":
        d, t = get_arrays("datasets/monk/monk.csv", "class")
        dataset = ds(data=d, target=t)
        inputs = 6
        outputs = 1

elif dataset_name == "penguins":
        d, t = get_arrays("datasets/penguins/penguins.csv", "species")
        dataset = ds(data=d, target=t)
        inputs = 6
        outputs = 3

elif dataset_name == "alcohol":
        d, t = get_arrays("datasets/alcohol/QCM3.csv", "class", sep=";")
        dataset = ds(data=d, target=t)
        inputs = 10
        outputs = 5

elif dataset_name == "vehicle":
        d, t = get_arrays("datasets/vehicle/vehicle.dat", "Column18", first_row=True, sep=" ")
        dataset = ds(data=d, target=t)
        inputs = 18
        outputs = 4

elif dataset_name == "breast_cancer":
        d, t = get_arrays("datasets/breast/breast-cancer-wisconsin.data", "Column10", first_column=True, first_row=True)
        dataset = ds(data=d, target=t)
        inputs = 9
        outputs = 1