from flask import Flask, render_template, request
import joblib
import re
from konlpy.tag import Okt
import pandas as pd
#플라스크 관련 설정
app = Flask(__name__)
app.debug =True

okt=Okt()
model_nb = None
model_lr = None
tfidf_vector=None
dtm_vector=None
#바인딩
def load_lr():
    global model_lr, tfidf_vector
    model_lr = joblib.load("model/model_lr.pkl")
    tfidf_vector = joblib.load("model/model_lr_dtm.pkl")
def load_nb():
    global model_nb, dtm_vector
    model_nb = joblib.load("model/model_nb.pkl")
    dtm_vector = joblib.load("model/model_nb_dtm.pkl")
#토크나이저
def tw_tokenizer(text):
    token_ko = okt.morphs(text)
    return token_ko
def lt_t(text):
    review = re.sub(r"\d+", "", text)
    text_vector = tfidf_vector.transform([review])
    return text_vector
def lt_nb(text):
    stopwords=["은","는","이","가"]
    review = text.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]","")
    morphs=okt.morphs(review, stem=True)#토큰 분리
    test = " ".join(morph for morph in morphs if not morph in stopwords)
@app.route("/")
def index():
    menu ={
        "home": True,
        "senti":False
    }
    return render_template("home.html", menu=menu)
@app.route("/senti", methods=['GET', 'POST'])
def senti():
    menu ={
        "home": False,
        "senti":True
    }
    if request.method =="GET":
        return render_template("senti.html", menu=menu) 
    else:
        review = request.form["review"]
        review_text = lt_t(review)
        lr_result = model_lr.predict(review_text)[0]
        nb_result = model_lr.predict(review_text)[0]
        lr="긍정" if lr_result else "부정"
        nb="긍정" if lr_result else "부정"
        movie = {"review":review, "lr":lr, "nb":nb}
        return render_template("senti_result.html", menu=menu, movie=movie)
if __name__ == '__main__':
    load_lr()
    load_nb()
    app.run()
