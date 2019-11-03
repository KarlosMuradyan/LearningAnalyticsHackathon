from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import scipy.spatial.distance as dist
import pickle
import json

app = Flask(__name__)
initial_df = pd.read_csv('/home/karlos/Documents/workspace/LearningAnalyticsHackathon/ubc_course_calendar_data.csv')
df = initial_df[initial_df.COURSE_DESCRIPTION.notnull()]

df = df[df['COURSE_DESCRIPTION'] != ' ']
unique_course_descriptions = df['COURSE_DESCRIPTION'].unique()

with open('../src/big_dict.pkl', 'rb') as handle:
    description_to_courses = pickle.load(handle)

with open('../src/small_dict.pkl', 'rb') as handle:
    vector_to_description = pickle.load(handle)

with open('../src/faculty_dict.json', 'rb') as f:
    faculty_to_description = json.load(f)   



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/get_courses', methods=['POST'])
def predict():
    faculty_of_interest = request.form.get('Faculty').upper()
    if faculty_of_interest:
        list_of_descriptions = np.array(faculty_to_description.get(faculty_of_interest, [-1]))
    else:
        list_of_descriptions = [] 
        for val in faculty_to_description.values(): 
            list_of_descriptions.extend(val)
        list_of_descriptions = np.array(list_of_descriptions)
    print('list of descriptions::: ')
    print(len(list_of_descriptions))
    print(list_of_descriptions[:5])

    if list_of_descriptions.__len__() == 1 and list_of_descriptions[0] == -1:
        return [-1]

    list_of_vectors = np.array([description_to_courses[description]['vector'] for description in list_of_descriptions])

    with app.graph.as_default():
        # sess = tf.Session(graph = app.graph)
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.tables_initializer())
        input_sentence = request.form.get('Sentence')
        top = int(request.form.get('Top'))

        my_vector = app.sess.run(app.embed_model([input_sentence]))
    # sess.close()
    # sess.close()
    print('Calculating the distances: ')
    all_distances = np.array([dist.cityblock(my_vector[0], vector) for vector in list_of_vectors])
    print(all_distances[:5])

    print('Sorting and giving results: ')
    indices = all_distances.argsort()[:top]
    print(indices)
    print('First 5 descriptions from the list ')
    print(list_of_descriptions[:5])

    print('The size of the list of descriptions ')
    print(len(list_of_descriptions))

    print('Type of list of descriptions')
    print(type(list_of_descriptions))

    print('Get shape of the list of desc')
    print(list_of_descriptions.shape)
    best_matching_descriptions = list_of_descriptions[indices]

    print(f'The input is: {input_sentence}')
    print(f'Top {top} matching sentences are: ')
    print(best_matching_descriptions)

    courses_matching = []

    print('Courses matching the descrition')
    for matching_desc in best_matching_descriptions:

        for i in range(len(description_to_courses[matching_desc]['courses'])):
            description_to_courses[matching_desc]['courses'][i]['description'] = matching_desc
        courses_matching.extend(description_to_courses[matching_desc]['courses'])

    print(courses_matching)
    result = {'matching_courses' : courses_matching}

    return jsonify(result)



@app.before_first_request
def load_model_to_app():

    print('Shape of the unique course descriptions: ', len(unique_course_descriptions))

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    # embeddings = embed(unique_course_descriptions)
    # input_string = embed(['Organize seminars and workshops'])


    print('Starting the tensorflow session')
    app.graph = tf.get_default_graph()
    all_distances = []
    with app.graph.as_default():
        sess = tf.Session(graph = app.graph).__enter__()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # embed_vectors = sess.run(embeddings)
    # sess.close()
    app.sess = sess

    # string_to_vector_dict = dict(zip(unique_course_descriptions, embed_vectors))

    app.embed_model = embed
    # app.embed_vectors = embed_vectors



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
