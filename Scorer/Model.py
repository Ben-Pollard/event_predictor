import numpy as np

class model:
    def __init__(self, rfr_search, memberships_per_group, maxRsvpLimit, w2vModel, vecSize):
        self.rfr_search = rfr_search
        self.memberships_per_group = memberships_per_group
        self.maxRsvpLimit = maxRsvpLimit
        self.w2vModel = w2vModel
        self.vecSize = vecSize

    def vectorise_topics(self, group_topics):
        wvecs = []
        for topic in group_topics:
            try:
                wvecs.append(self.w2vModel.wv[topic])
            except:
                None
        result = np.vstack(wvecs).mean(axis=0) if len(wvecs) > 0 else np.zeros(self.vecSize)
        return result

    def score(self, query):
        record_to_score = np.hstack([np.array([query['rsvp_limit'] if query['rsvp_limit'] else self.maxRsvpLimit,
                                               query['duration'] if query['duration'] else -1,
                                               query['time'] - query['created'],
                                               self.memberships_per_group[query['group_id']]]),
                                     self.vectorise_topics(query['topics'])]).reshape(1, -1)
        prediction = int(round(self.rfr_search.best_estimator_.predict(record_to_score)[0]))
        return prediction