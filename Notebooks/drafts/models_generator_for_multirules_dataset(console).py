import pandas as pd
import sweetviz as sv
from pycaret.classification import *


def read_dataset():
    data = pd.read_csv('./data/50901statsFull34_double.csv', sep=';')

    records = data["ViolatedRule"].str.split(";", n = 1, expand = True)
    for i in range(records.shape[1]):
        data[f"ViolatedRule{i}"]= records[i]
        data.loc[data[f"ViolatedRule{i}"] != "-1", f"ViolatedRule{i}"] = 1
        data.loc[data[f"ViolatedRule{i}"] == "-1", f"ViolatedRule{i}"] = 0

    data = data.drop(columns=['txId', 'FeatureID', 'ViolatedRule'])

    data = data.astype({'LeftSieFirst': 'float', 'LeftSieLst': 'float',
                       'RightSieFirst': 'float', 'RightSieLst': 'float', 'Length': 'float'})


    data['left_diff'] = data['LeftSieFirst'] - data['LeftSieLst']
    data['right_diff'] = data['RightSieFirst'] - data['RightSieLst']
    data['left_dens'] = data['left_diff'] / data['Length']
    data['right_dens'] = data['right_diff'] / data['Length']
    return data


def prepare_eda(data):
    # skip=["proline", "magnesium"],
    config = sv.FeatureConfig(force_num=['ViolatedRule'])
    my_report = sv.analyze(data, feat_cfg=config, target_feat='ViolatedRule')
    my_report.show_html()
    # profile = ProfileReport(data, title="Pandas Profiling Report")
    # profile.to_file(output_file=pathlib.Path("./data_report.html"))
    # profile.to_widgets()


def stack():
    top3 = compare_models(n_select=3)
    tuned_top3 = [tune_model(i) for i in top3]
    blender = blend_models(tuned_top3)
    stacker = stack_models(tuned_top3)
    best_auc_model = automl(optimize='AUC')
    print(best_auc_model)



if __name__ == '__main__':
    data = read_dataset()
    # prepare_eda(data)

    print(data.info())
    print(data.describe().transpose())


    for column in data.columns.values:
        if column.startswith('ViolatedRule'):
            print(data[column].value_counts())

            s = setup(data, target=column, silent=True, log_experiment=False, experiment_name='first_rule')
            # best = compare_models(turbo=True)
            # results = pull()
            # print(results.head())
            investigated = create_model('xgboost')

            # models()
            print(investigated)
            #deep_check(investigated)
            tuned_investigated = tune_model(investigated, choose_better=True, n_iter=10, search_library='optuna', search_algorithm='tpe')
            print(tuned_investigated)
            # plot_model(tuned_investigated, plot='pr')
            # plot_model(investigated, plot='feature')
            # plot_model(investigated, plot='confusion_matrix')
            print(interpret_model(tuned_investigated))
            # evaluate_model(tuned_investigated)
            # predict_model(tuned_investigated)
            # final_rf = finalize_model(tuned_investigated)
