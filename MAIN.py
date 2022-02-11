from flask import Flask, abort
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
import json
import string
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
new_words = ['another', 'app', 'able', 'computer', 'customer', 'dashlane', 'desktop', 'didnt', 'get', 'good', 'got',
             'hello', 'help', 'hi', 'issue', 'keep', 'know', 'knowuse', 'many', 'need', 'new', 'now', 'one', 'password',
             'passwordapp', 'passwords', 'person', 'please', 'problem', 'product', 'resolved', 'said', 'seems',
             'solution', 'still', 'support', 'thank', 'thanks', 'thing', 'time', 'tried', 'use', 'using', 'want', 'wanted',
             'will', 'work', 'working', 'you', 'would', 'cant', 'email', 'month', 'like', 'never', 'took', 'dont', 'even',
             'helpful', 'difficult', 'none', 'really','info', 'best', 'perfekt', 'great']
stopword.extend(new_words)

parser = reqparse.RequestParser()
parser.add_argument('data',required=True)
parser.add_argument('msgs',required=True)
parser.add_argument('msgid')
parser.add_argument('numberoftopics',type=int,required=True)
parser.add_argument('extrastopwords')

app = Flask(__name__)
api = Api(app)
CORS(app)


def doNone(df):
    for j in df.columns:
        for i in range(len(df[j])):
            if df[j][i] == '':
                df[j][i] = None


def cleantext(text):
    text1 = ''.join([i for i in text if i not in string.punctuation])
    text2 = ''.join([i for i in text1 if not i.isdigit()])
    text3 = [i for i in text2.split() if i.lower() not in stopword]
    return text3

class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}

class wordclouds(Resource):
    def post(self):
        resulttosend = {}
        arg = parser.parse_args()
        data = json.loads(arg['data'])
        topics = int(arg['numberoftopics'])
        newcolumns = ['feedback', 'userid']
        try:
            df = pd.DataFrame(data)
        except:
            abort(404, description='Dataset should be as required')
        oldcolumns = df.columns
        if len(oldcolumns) == len(newcolumns):
            Dict = {}
            for i in range(len(newcolumns)):
                Dict[oldcolumns[i]] = newcolumns[i]
            df = df.rename(columns=Dict)
        else:
            abort(
                404, description='No need for extra data just feedback and userid remove other coloumns')
        doNone(df)
        df = df.dropna(axis=0)
        df.reset_index(inplace=True)
        for i in range(len(df['feedback'])):
            df['feedback'][i] = ' '.join(cleantext(df['feedback'][i]))
        vectorizer = TfidfVectorizer(
            max_df=0.94, min_df=2)
        x = vectorizer.fit_transform(df['feedback'])
        model = NMF(n_components=topics, random_state=42)
        model.fit(x)
        nmf_features = model.transform(x)
        nmf_features.argmax(axis=1)
        df['Topic'] = nmf_features.argmax(axis=1)
        if arg['msgs'] == 'True':
            allmsg = []
            if len(arg['extrastopwords']) > 0:
                for i in arg['extrastopwords']:
                    stopword.append(i.lower())
            for i in range(topics):
                df1 = df[df['Topic'] == i]
                feedback = []
                for j in df1['feedback']:
                    feedback.append(j)
                allmsg.append(feedback)
            resulttosend['msgs'] = allmsg
        if arg['msgid'] == 'True':
            Userid = []
            for i in range(topics):
                df1 = df[df['Topic'] == i]
                ids = []
                for j in df1['userid']:
                    ids.append(j)
                Userid.append(ids)
            resulttosend['Ids'] = Userid
        return resulttosend

api.add_resource(status, '/')
api.add_resource(wordclouds, "/wordclouds")

if __name__ == '__main__':
    app.run()
