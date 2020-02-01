from os import path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sta211.datasets import load_train_dataset, load_test_dataset, find_best_train_dataset
from sklearn.model_selection import GridSearchCV
from sta211.selection import get_naive_bayes, get_mlp, get_svm, get_gradient_boosting, get_random_forest, get_best_hyper_parameters, get_extra_trees, get_adaboost, get_voting_classifier
from multiprocessing import cpu_count


n_jobs = max(1, cpu_count()-1)
test_size = 0.20

X, y, quantitatives = load_train_dataset()

# Manual aggregation
pipe, search_grid = get_voting_classifier()

# pipes, search_grid = get_svm()
# pipe = Pipeline(pipes)

cv = StratifiedShuffleSplit(test_size=test_size, random_state=0, n_splits=5)
grid = GridSearchCV(pipe, search_grid, cv=cv, n_jobs=n_jobs, return_train_score=True, refit=True, scoring="accuracy")
grid.fit(X, y)

parameters = get_best_hyper_parameters(grid)
print("Result for {} configurations".format(len(parameters)))
for p in parameters:
    print("{};{:.2f}%;{:.4f}%;±{:.4f}%".format(
        ", ".join(map(lambda k: "{}={}".format(k.split("__")[1], p["params"][k]), p["params"].keys())),
        100.0 * p["mean_train_score"],
        100.0 * p["mean_test_score"],
        200.0 * p["std_test_score"]
    ))

    # print("Results: Train: {:.2f}%, Test: {:.2f}% std:{:.4f} for {}".format(100 * p["mean_train_score"], 100 * p["mean_test_score"], p["std_test_score"], p["params"]))

prediction_file = "{}/predictions.csv".format(path.dirname(path.abspath(__file__)))
pred = grid.predict(load_test_dataset())
f = open(prediction_file, "w")
f.write("\n".join(map(lambda o: str(o), pred)))
f.close()
