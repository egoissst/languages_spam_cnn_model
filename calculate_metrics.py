import pandas as pd

def get_pred_cnt_file(file_loc, spam_threshold):
    """
    returns positive and negative predicted count from the prediction file
    """
    prob_field_name = 'model_1579177474_MALAY'
    df_file = pd.read_csv(file_loc)
    pos_pred_cnt = sum(df_file[prob_field_name]>spam_threshold)
    neg_pred_cnt = sum(df_file[prob_field_name]<spam_threshold)
    return (pos_pred_cnt, neg_pred_cnt)


def get_metric_values(spam_threshold, spam_file, ham_file):
    """
    prints metric values like precision and recall for the given value of the spam threshold, spam file and the ham file location
    """
    true_positive, false_negative = get_pred_cnt_file(spam_file, spam_threshold)
    false_positive, true_negative = get_pred_cnt_file(ham_file, spam_threshold)

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = (true_positive) / (true_positive + false_positive)
    recall = (true_positive) / (true_positive + false_negative)

    print('SPAM THRESHOLD : {}'.format(spam_threshold))
    print('accuracy : {}'.format(accuracy))
    print('precision : {}'.format(precision))
    print('recall : {}'.format(recall))


BASE_FILE_LOC = "/data1/language_spam_model/cnn_text_classification_cahya/"
spam_file = BASE_FILE_LOC + "saved_models/model_1579177474_MALAY/malay_spam_16Jan_pos_prediction.csv"
ham_file = BASE_FILE_LOC + "saved_models/model_1579177474_MALAY/malay_ham_16Jan_neg_prediction.csv"

#SPAM_THRESHOLD = 0.5
#spam_threshold_array = [x/10 for x in range(5,10,1)]
spam_threshold_array = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

for spam_threshold_val in spam_threshold_array:
    get_metric_values(spam_threshold_val, spam_file, ham_file)
    print()
