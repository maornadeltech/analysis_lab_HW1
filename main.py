import data_loader
import model
from sklearn.ensemble import RandomForestClassifier
import pickle


def run_main():
    data = data_loader.data_preprocess(num_rows=25)
    
    model.find_best_model(data)
    model.test_log_reg(data)
    model.test_null_models(data)

    best_model = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced')
    pickle.dump(best_model.fit(data['train'][0], data['train'][1]), open('/home/student/data/model.pkl', 'wb'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
