from sklearn.metrics import precision_recall_curve as pr_curve
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging


def plot_pr(name, labels, predictions, x_lower=0.0, x_upper=1.0, y_lower=0.0, y_upper=1.05, **kwargs):
    precision, recall, thresholds = pr_curve(labels, predictions)
    plt.plot(recall, precision, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([y_lower, y_upper])
    plt.xlim([x_lower, x_upper])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    return precision, recall, thresholds


def test_plot_pr(model, file_name, test_dataloader, x_lower=0.0, x_upper=1.0, y_lower=0.0, y_upper=1.05, output_dim=1):
    test_predictions = []
    test_labels = []
    for images, labels in test_dataloader:
        test_labels.append(labels[0])
        logits = model(images)
        predictions = tf.nn.softmax(logits, axis=1)[0, 1:]
        test_predictions.append(predictions[0])

    precision, recall, thresholds = plot_pr("Test Precision-Recall Curve", test_labels,
                                            test_predictions, x_lower, x_upper, y_lower, y_upper)

    plt.savefig(file_name, dpi=600)

    return precision, recall, thresholds


def test_plot_pr_segmentation(model, file_name, test_dataloader, x_lower=0.0, x_upper=1.0, y_lower=0.0, y_upper=1.05, output_dim=1):
    test_predictions = []
    test_labels = []
    for nonlenses, nonlenses_labels, lenses, lenses_labels, _ in test_dataloader:
        images = [nonlenses, lenses]
        labels = [nonlenses_labels, lenses_labels]
        for i in range(2):
            if output_dim == 1:
                test_labels.append(labels[i][0])
                predictions = model(images[i])[0]
                test_predictions.append(predictions)
            else:
                test_labels.append(labels[i][0])
                predictions = tf.nn.softmax(model(images[i]), axis=1)[0, 1:]
                test_predictions.append(predictions)
    precision, recall, thresholds = plot_pr("Test Precision-Recall Curve", test_labels,
                                            test_predictions, x_lower, x_upper, y_lower, y_upper)

    plt.savefig(file_name, dpi=600)

    return precision, recall, thresholds


def kfold_plot_pr(file_name, labels, predictions):
    plot_pr("Test Precision-Recall Curve",  labels, predictions)
    plt.savefig(file_name, dpi=600)


def threshold_using_precision(precision, recall, thresholds, precision_value):
    bools = (precision >= precision_value)
    precision_val = np.min(precision[bools])
    index = np.argmax(precision == precision_val)
    recall_val = recall[index]
    threshold_val = thresholds[index]
    return precision_val, recall_val, threshold_val


def threshold_using_recall(precision, recall, thresholds, recall_value):
    bools = (recall >= recall_value)
    recall_val = np.min(recall[bools])
    index = np.argmax(recall == recall_val)
    precision_val = precision[index]
    threshold_val = thresholds[index]
    return precision_val, recall_val, threshold_val


def recall_using_precision(precision, recall, thresholds, precision_value):
    bools = (precision >= precision_value)
    precision_val = np.min(precision[bools])
    index = (precision == precision_val)
    recall_val = np.mean(recall[index])
    #threshold_val = thresholds[index]
    return precision_val, recall_val,  # threshold_val


def precision_using_recall(precision, recall, thresholds, recall_value):
    bools = (recall >= recall_value)
    recall_val = np.min(recall[bools])
    index = (recall == recall_val)
    precision_val = np.mean(precision[index])
    #threshold_val = thresholds[index]
    return precision_val, recall_val,  # threshold_val
