# src/utils/utils.py

def filter_by_no_value(texts, labels, no_values, threshold=0.8):
    filtered_texts = []
    filtered_labels = []

    for text, label, no_val in zip(texts, labels, no_values):
        if no_val <= threshold:
            filtered_texts.append(text)
            filtered_labels.append(label)

    return filtered_texts, filtered_labels