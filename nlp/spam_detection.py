import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
import string

# 1. Load dữ liệu mẫu (có thể tải từ: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                 sep='\t', names=['label', 'message'])

# 2. Tiền xử lý nhãn
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 3. Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# 4. Chuyển văn bản thành vector bằng TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Dự đoán và đánh giá
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 7. Dự đoán tin nhắn mới
def predict_spam(message):
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    return "SPAM" if pred else "HAM"

# Thử dự đoán
print(predict_spam("Win a free iPhone now! Click here"))
print(predict_spam("Hey, are we still meeting at 3pm?"))
