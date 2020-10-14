import csv
import os.path
if __name__ == '__main__':
    all_data_dir = './dataset_all_phones_cut'
    model_dir = './pbtxt'
    phones_dir = os.listdir(all_data_dir)
    for phone_dir in phones_dir:
        phone_path = os.path.join(all_data_dir, phone_dir)
        labels_dir = os.listdir(phone_path)
        for label_dir in labels_dir:
            model_name = 'model_MI8_iphone7_' + label_dir
            predict_model = os.path.join(model_dir, model_name)
            predict_data = os.path.join(phone_path, label_dir)
            