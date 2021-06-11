import pandas as pd
import joblib
from config_folder.config import Config

config = Config.get_instance().config

model_path = "/home/sowmya_kanuri/PycharmProjects/receipt_transaction_matching/saved_models/model_1623395205.0765216"

path_to_test_set = config['path_to_test_file']
test_set = pd.read_csv(path_to_test_set)

filtered_test_set = test_set.drop(["receipt_id","company_id","matched_transaction_id","feature_transaction_id"],
                                  axis=1,inplace=False)

loaded_model = joblib.load(model_path)

matching_scores = loaded_model.predict_proba(filtered_test_set)[:,1:2]

matching_scores = list(matching_scores.reshape((1,matching_scores.shape[0]))[0])

test_set["matching_scores"] = matching_scores

test_set.sort_values(by=["matching_scores"])

test_set.drop(["matching_scores"],axis=1)

path_to_save_test_results = config["path_to_save_test_results"] + str("ordered_transactions.csv")

test_set.to_csv(path_to_save_test_results)

print(matching_scores)