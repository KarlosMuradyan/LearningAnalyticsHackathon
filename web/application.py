from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import scipy.spatial.distance as dist
import pickle

app = Flask(__name__)
initial_df = pd.read_csv('/home/karlos/Documents/workspace/LearningAnalyticsHackathon/ubc_course_calendar_data.csv')
df = initial_df[initial_df.COURSE_DESCRIPTION.notnull()]

df = df[df['COURSE_DESCRIPTION'] != ' ']

unique_course_descriptions = df['COURSE_DESCRIPTION'].unique()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/get_courses', methods=['POST'])
def predict():
    print('INput SENTENCE ISSSS -?')
    print(request.form.get('Sentence'))

    print('GIVE TOPPP -?')
    print(request.form.get('Top'))

    with app.graph.as_default():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        input_sentence = request.form.get('Sentence')
        top = int(request.form.get('Top'))

        my_vector = sess.run(app.embed_model([input_sentence]))

        print('Calculating the distances: ')
        all_distances = np.array([dist.cityblock(my_vector[0], vector) for vector in app.embed_vectors])


        print('Sorting and giving results: ')
        indices = all_distances.argsort()[:top]
        best_matching_descriptions = unique_course_descriptions[indices]
        sess.close()
    print(f'The input is: {input_sentence}')
    print(f'Top {top} matching sentences are: ')
    print(best_matching_descriptions)

@app.before_first_request
def load_model_to_app():


    print('Shape of the unique course descriptions: ', len(unique_course_descriptions))

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    embeddings = embed(unique_course_descriptions)
    input_string = embed(['Organize seminars and workshops'])


    print('Starting the tensorflow session')
    app.graph = tf.get_default_graph()
    all_distances = []
    sess = tf.Session(graph = app.graph).__enter__()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    embed_vectors = sess.run(embeddings)
    sess.close()

    string_to_vector_dict = dict(zip(unique_course_descriptions, embed_vectors))

    app.embed_model = embed
    app.embed_vectors = embed_vectors



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
