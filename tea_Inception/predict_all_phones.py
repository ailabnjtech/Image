# import cv2
import tensorflow as tf
import os.path
import numpy as np
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def predict(predict_data, predict_model):
    strings = ['true', 'false']

    def id_to_string(node_id):
        return strings[node_id]

    tf.reset_default_graph()
    with tf.gfile.FastGFile(predict_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    count_0 = 0
    count_1 = 0
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('output/prob:0')
        # 遍历目录
        for root, dirs, files in os.walk(predict_data):
            for file in files:
                # 载入图片
                image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()

                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
                predictions = np.squeeze(predictions)  # 把结果转为1维数据

                # 打印图片路径及名称
                image_path = os.path.join(root, file)
                # print(image_path)

                # 排序
                top_k = predictions.argsort()[::-1]
                if top_k[0] == 0:
                    count_0 += 1
                if top_k[1] == 0:
                    count_1 += 1
                # print(top_k)
                # for node_id in top_k:
                #     # 获取分类名称
                #     human_string = id_to_string(node_id)
                #     # 获取该分类的置信度
                #     score = predictions[node_id]
                #     print('%s (score = %.5f)' % (human_string, score))
    if count_1 > count_0:
        accuracy = '%.1f%%' % ((count_1 / (count_1 + count_0)) * 100)
    else:
        accuracy = '%.1f%%' % ((count_0 / (count_1 + count_0)) * 100)
    # print(predict_model, predict_data)
    # print(count_0, count_1)
    # print(accuracy)
    csv_data[count].append(accuracy)
    #             img = cv2.imread(image_path)
    #             cv2.imshow('image', img)
    #             cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    all_data_dir = './dataset_all_phones_cut'
    model_dir = './pbtxt'
    csv_data = [[] for i in range(450)]
    count = 0
    no_label = []
    labels = ['1', '2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
              '21', '22', '23', '24', '25', '27', '28', '29', '30', '31', '34', '35', '36', '38', '39', '40', '42',
              '43', '51', '52', '59', '60', '61', '62', '63', '64']
    phones_dir = os.listdir(all_data_dir)
    for phone_dir in phones_dir:
        phone_path = os.path.join(all_data_dir, phone_dir)
        labels_dir = os.listdir(phone_path)

        for label_dir in labels_dir:
            if label_dir not in labels:
                no_label.append(label_dir)
                continue
            model_name = 'model_MI8_iphone7_' + label_dir
            predict_model = os.path.join(model_dir, model_name)
            predict_data = os.path.join(phone_path, label_dir)
            csv_data[count].append(phone_dir)
            csv_data[count].append(label_dir)
            predict(predict_data, predict_model)
            count += 1

    print(csv_data)
    print(no_label)
    with open('./predict_accuracy.csv', 'a+', newline='', ) as csvfile:
        writer = csv.writer(csvfile)
        for row in csv_data:
            writer.writerow(row)
    print('########################################################################')
