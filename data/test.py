import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# **读取 TXT 语料库**
def load_corpus(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# **计算 TF-IDF**
def compute_tfidf(corpus, save_path="tfidf_model.pkl"):
    vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # **保存模型**
    with open(save_path, 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix), f)

    print(f"✅ TF-IDF 模型已保存到: {save_path}")
    return vectorizer, tfidf_matrix

# **运行**
#txt_file = "D:/SimCSE-main/SimCSE-main/data/wiki1m_for_simcse.txt"
#corpus = load_corpus(txt_file)
#vectorizer, tfidf_matrix = compute_tfidf(corpus, "tfidf_model2.pkl")
# **加载 TF-IDF 模型**
def load_tfidf_model(model_path):
    with open(model_path, 'rb') as f:
        vectorizer, tfidf_matrix = pickle.load(f)
    print("✅ TF-IDF 模型已加载")
    return vectorizer, tfidf_matrix

# **提取某个句子的关键词**
def extract_keywords(sentence, vectorizer, top_k=5):
    feature_names = vectorizer.get_feature_names_out()
    sentence_tfidf = vectorizer.transform([sentence]).toarray()[0]
    
    # 取出最高分的关键词
    sorted_keywords = sorted(zip(feature_names, sentence_tfidf), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:top_k]]

# **加载模型并查询**
vectorizer, tfidf_matrix = load_tfidf_model("tfidf_model.pkl")

sentence = "Artificial intelligence and deep learning are widely used in industry."
keywords = extract_keywords(sentence, vectorizer, top_k=3)

print("🔍 关键词:", keywords)
