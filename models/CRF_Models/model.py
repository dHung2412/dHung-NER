# file: model.py

import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics

class CRFModel:
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True):
        self.params = {
            'algorithm': algorithm,
            'c1': c1,
            'c2': c2,
            'max_iterations': max_iterations,
            'all_possible_transitions': all_possible_transitions
        }
        self.model = sklearn_crfsuite.CRF(**self.params)

    def fit(self, X_features, y_labels):
        print("Bắt đầu huấn luyện CRF...")
        print(f"Params: {self.params}")
        self.model.fit(X_features, y_labels)
        print("--- Huấn luyện hoàn tất ---")

    def predict(self, X_features):
        return self.model.predict(X_features)

    def evaluate(self, X_features, y_true, labels):
        """Đánh giá và in báo cáo"""
        y_pred = self.predict(X_features)
        
        print(metrics.flat_classification_report(
            y_true, y_pred, labels=labels, digits=4
        ))
        return y_pred

    def save(self, filepath):
        """Lưu model đã huấn luyện"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Mô hình đã được lưu tại {filepath}")

    @staticmethod
    def load(filepath):
        """Tải model đã huấn luyện"""
        with open(filepath, 'rb') as f:
            crf_instance = pickle.load(f)
        
        # Tạo một wrapper mới và gán model đã tải vào
        model_wrapper = CRFModel()
        model_wrapper.model = crf_instance
        # Ghi đè params (mặc dù chúng không được dùng sau khi tải)
        model_wrapper.params = crf_instance.get_params()
        print(f"Mô hình đã được tải từ {filepath}")
        return model_wrapper