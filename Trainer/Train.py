import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.externals import joblib
import itertools
from gensim.models import Word2Vec
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


def vectorise_topics(group_topics, word2vecModel, vecSize):
    """Generates a word vector from a word2vec model
    :param group_topics: A list of topic strings
    :param word2vecModel: A trained instance of gensim.models.Word2Vec of size vecSize
    :param vecSize: The length of the vector the model has been trained to return
    :return: A numpy array representing the word vector
    """
    wvecs = []
    for topic in group_topics:
        try:
            wvecs.append(word2vecModel.wv[topic])
        except:
            None
    result = np.vstack(wvecs).mean(axis=0) if len(wvecs) > 0 else np.zeros(vecSize)
    return result


def vectorise_topics_vectorised(events, word2vecModel, vecSize):
    """
    Applies vectorise_topics to a dataframe
    :param events: A pandas.DataFrame that includes a column called 'topics' where
    each cell contains a list of topic strings
    :param word2vecModel: A trained instance of gensim.models.Word2Vec of size vecSize
    :param vecSize: The length of the vector the model has been trained to return
    :return: A copy of the dataframe with new fields topicvec_0 .. topicvec_n
    """
    vectorised_topics = np.vstack(events.topics.apply(lambda x: vectorise_topics(x, word2vecModel, vecSize)))
    for idx in range(vectorised_topics.shape[1]):
        events['topicvec_' + str(idx)] = vectorised_topics[:, idx]
    return events


def evaluate(model, X_train, X_test, Y_train, Y_test):
    """
    Displays evaluation of a scikit-learn model using the R-squared metric
    :param model: A trained scikit-learn model
    :param X_train: Dataframe containing the model features (training split)
    :param X_test: Dataframe containing the model features (testing split)
    :param Y_train: Series containing the target variable (training split)
    :param Y_test: Series containing the target variable (testing split)
    """
    print("Parameters of best model:\n %s" % model.get_params())

    r2_train = r2_score(Y_train, model.predict(X_train))
    r2_test = r2_score(Y_test, model.predict(X_test))
    print("\nR-square train: %f" % r2_train)
    print("R-square test: %f" % r2_test)

    if type(model) == RandomForestRegressor:
        print("\n")
        print(pd.DataFrame({'features': X_train.columns, 'importances': model.feature_importances_}). \
              sort_values('importances', ascending=False))

    return None


def parse_rsvp(rsvp_json):
    """
    Parses the number of positive RSVPs
    :param rsvp_json: List of dicts that include a 'response' key
    :return: Count of positive responses
    """
    return sum([e['response']=='yes' for e in rsvp_json])




def train(save_to):
    """
    Train a model to predict the number of attendees of a tech meetup.

    Train a model and then serialise all objects necessary for scoring new data.
    :param save_to: String representing the location to pickle the scoring objects to.
    :return:
    """

    #Get raw data
    events = pd.read_json('Data/events.json')
    groups = pd.read_json('Data/groups.json')
    users = pd.read_json('Data/users.json')
    print("Number of events: %i" % len(events))
    print("Number of events in past: %i" % (events.status == 'past').sum())


    #Create target variable
    events['rsvp_yes'] = events.rsvps.apply(lambda x: parse_rsvp(x))

    ###Create feature 'rsvp_limit'
    maxRsvpLimit = events.rsvp_limit.max()
    events['rsvp_limit'].fillna(maxRsvpLimit, inplace=True)

    ###Create feature 'duration'
    events['duration'].fillna(-1, inplace=True)

    ###Create feature 'group_member_count'
    #Get memberships of users
    memb_joined = users.memberships.apply(lambda membs: [m['joined'] for m in membs]).values
    memb_group_id = users.memberships.apply(lambda membs: [m['group_id'] for m in membs]).values
    memb_user = users.apply(lambda row: [row.user_id for m in row.memberships], axis=1).values
    memberships = pd.DataFrame({'user_id':list(itertools.chain.from_iterable(memb_user)),
                               'joined':list(itertools.chain.from_iterable(memb_joined)),
                               'group_id':list(itertools.chain.from_iterable(memb_group_id))})

    #This is used for scoring
    memberships_per_group = memberships.groupby('group_id')['user_id'].count().sort_values(ascending=False)

    #Join group memberships to event data
    events['event_id'] = events.index
    events_with_users = events[['event_id','group_id','created']].merge(memberships)
    events_with_users[events_with_users.created > events_with_users.joined]
    group_member_count = events_with_users.groupby('event_id')['group_id'].count()
    events['group_member_count'] = group_member_count

    ###Create feature 'timeFromCreationToEvent'
    events['timeFromCreationToEvent'] = events.time - events.created

    ###Create feature 'topics'
    events['topics'] = events.merge(groups, on='group_id').sort_values('event_id').reset_index().topics


    ###Create training and testing data
    fields = ['rsvp_limit','duration','timeFromCreationToEvent','group_member_count', 'topics']
    selected_events = events[events.status == 'past'][fields + ['rsvp_yes']]

    X = selected_events.drop('rsvp_yes', axis=1)
    Y = selected_events['rsvp_yes']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.75)
    print("%i records in the training set" % Y_train.count())
    print("%i records in the testing set" % Y_test.count())

    #Train Word2Vec model on training data then apply it to training and testing sets
    vecSize = 5
    w2vModel = Word2Vec(X_train.topics, size=vecSize, min_count=1)
    X_train = vectorise_topics_vectorised(X_train, w2vModel, vecSize).drop('topics', axis=1)
    X_test = vectorise_topics_vectorised(X_test, w2vModel, vecSize).drop('topics', axis=1)


    ###Train and evaluate model
    param_dist_rf = {"max_depth": [3, None],
                  #"max_features": randint(1, len(X_train.columns)),
                  "min_samples_split": randint(2, 11),
                  "min_samples_leaf": randint(1, 11),
                  "n_estimators": randint(1, 100),
                  "bootstrap": [True, False]}
    rfr = RandomForestRegressor()
    rfr_search = RandomizedSearchCV(rfr, param_distributions=param_dist_rf, n_iter=20)
    rfr_search.fit(X_train, Y_train)

    evaluate(rfr_search.best_estimator_, X_train, X_test, Y_train, Y_test)


    ###Save objects required for scoring
    required_fields = ['rsvp_limit','duration','time','created','group_id','topics']

    scorObj = {'rfr_search': rfr_search,
                'memberships_per_group': memberships_per_group,
                'maxRsvpLimit': maxRsvpLimit,
                'w2vModel': w2vModel,
                'required_fields': required_fields,
              'vecSize':vecSize}
    joblib.dump(scorObj, save_to)

    return None


if __name__ == '__main__':
    train(save_to = 'Data/scoring_objects.pkl')
