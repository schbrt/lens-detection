import sklearn.cluster as clus

def feat_list(images):
    feats = []
    for image in images:
        feat = image.features
        feats.append(feat)
    return feats

def cluster(train_data):
    data = []
    for item in train_data:
        for row in item:
            data.append(row)
    #change n_jobs to -1 to speed up (parallelize)
    km = clus.KMeans(n_clusters = len(data)/2, init ='k-means++', max_iter = 300, n_jobs = 1)
    km.fit(data)
    return km

def build_hist(data):
    hist = {}
    for item in data:
        if item in hist:
            hist[item] += 1
        else:
            hist[item] = 1

    return hist